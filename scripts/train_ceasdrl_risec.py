"""Train action and gold-action-conditioned argument cEASDRL on RISeC.

The verified GPU CNN uses original embeddings, binary reward, MSE Q targets,
and reward-sign replay sampling. Vectorized episodes and hyperparameters are
recorded explicitly. No ES/OP/EX or semantic-role classification is performed.
"""
import argparse
import collections
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from ceasdrl_torch import CnnDqn, Features, Replay, binary_rewards, dqn_loss, evaluate, scores


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--parity-report', type=Path, required=True)
    p.add_argument('--mode', choices=['act', 'arg'], required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--env-batch', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--train-frequency', type=int, default=4)
    p.add_argument('--target-updates', type=int, default=100)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--max-vector-steps', type=int, default=0, help='Engineering check only; 0 for full epochs')
    p.add_argument('--span-labels', action='store_true', help='Train/evaluate complete BIO spans instead of legacy first tokens')
    p.add_argument('--span-replay', choices=['upstream', 'balanced'], default='upstream')
    p.add_argument('--span-rate', choices=['tokens', 'mentions'], default='tokens')
    p.add_argument('--resume', action='store_true', help='Resume exactly from the last completed epoch, including replay and RNG')
    args = p.parse_args()
    if min(args.epochs, args.patience, args.env_batch, args.batch_size, args.train_frequency, args.target_updates) < 1:
        p.error('Counts must be positive')
    if not args.span_labels and (args.span_replay != 'upstream' or args.span_rate != 'tokens'):
        p.error('BIO adaptations require --span-labels')
    if args.resume and (args.output / 'receipt.json').exists():
        raise ValueError('Run is completed; refusing to overwrite its receipt by resuming')
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise ValueError('Refusing to overwrite run')
    if args.resume and not (args.output / 'last.pt').is_file():
        raise ValueError('No resumable checkpoint')
    model_class, feature_class, evaluator, reward_function, replay_class = CnnDqn, Features, evaluate, binary_rewards, Replay
    if args.span_labels:
        from ceasdrl_spans import SpanDqn, SpanFeatures, evaluate_spans, span_rewards, span_starts, BalancedSpanReplay
        model_class, feature_class, evaluator, reward_function = SpanDqn, SpanFeatures, evaluate_spans, span_rewards
        if args.span_replay == 'balanced':
            replay_class = BalancedSpanReplay
    parity = json.loads(args.parity_report.read_text(encoding='utf-8'))
    if not parity['passed']:
        raise ValueError('Keras/GPU parity check has not passed')
    if not torch.cuda.is_available():
        raise RuntimeError('This run requires the verified CUDA backend')
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
    start = time.time()
    features = feature_class(args.features, args.mode, device)
    train, dev, test = [features.examples[s] for s in ['train', 'dev', 'test']]
    groups = [set(x['doc_id'] for x in split) for split in [train, dev, test]]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError('Document split leakage')
    model = model_class(args.mode).to(device)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    target = copy.deepcopy(model).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-8)
    replay = replay_class()
    epoch_steps = sum(len(x['tokens']) for x in train)
    epsilon_decay = max(1000, epoch_steps)
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    configuration.update({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
                          'gamma': .9, 'epsilon_decay_steps': epsilon_decay,
                          'num_words': features.num_words, 'initialization': 'from scratch',
                          'task': ('Complete BIO span extraction' if args.span_labels else 'Binary occurrence extraction') + '; no ES/OP/EX',
                          'selection_metric': 'Dev exact span micro F1' if args.span_labels else 'Dev first-token micro F1',
                          'argument_task': ('Gold-action-conditioned explicit argument full spans, role agnostic' if args.span_labels else
                                            'Gold-action-conditioned explicit argument first tokens, role agnostic') if args.mode == 'arg' else None})
    total_steps = vector_steps = updates = 0
    best_f1, best_epoch, stale = -1., -1, 0
    history, visited, epoch_complete = [], set(), False
    first_epoch = 0
    if args.resume:
        saved = torch.load(args.output / 'last.pt', map_location=device, weights_only=False)
        old_config = saved['configuration']
        old_config.setdefault('span_replay', 'upstream')
        old_config.setdefault('span_rate', 'tokens')
        if not saved['history'][-1]['complete']:
            raise ValueError('Cannot resume a deliberately partial engineering-check epoch')
        if {k: v for k, v in configuration.items() if k != 'resume'} != {k: v for k, v in old_config.items() if k != 'resume'}:
            raise ValueError('Resume configuration changed')
        manifest_hash = hashlib.sha256((args.features / 'manifest.json').read_bytes()).hexdigest()
        if saved['feature_manifest_sha256'] != manifest_hash:
            raise ValueError('Resume dataset/features changed')
        model.load_state_dict(saved['state_dict'])
        target.load_state_dict(saved['target_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        replay = saved['replay']
        total_steps, vector_steps, updates = saved['training_steps'], saved['vector_steps'], saved['updates']
        best_f1, best_epoch, stale = saved['best_f1'], saved['best_epoch'], saved['stale']
        history, visited = saved['history'], set(saved['visited'])
        initial_dev, first_epoch = saved['initial_dev'], saved['epoch']
        if stale >= args.patience:
            first_epoch = args.epochs  # Finish evaluation after an interrupted early stop.
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    else:
        initial_dev, _ = evaluator(model, features, dev)
    write_json(args.output / 'config.json', configuration)
    print('Initial dev {}'.format(initial_dev), flush=True)
    last_report = time.time()
    stop = False
    for epoch in range(first_epoch, args.epochs):
        order = list(train)
        random.shuffle(order)
        losses, epoch_visited = [], set()
        epoch_start = time.time()
        model.train()
        for chunk_start in range(0, len(order), args.env_batch):
            chunk = order[chunk_start:chunk_start + args.env_batch]
            words, distance = features.batch(chunk)
            lengths = np.array([len(x['tokens']) for x in chunk])
            gold = np.zeros((len(chunk), features.num_words), dtype=np.int64 if args.span_labels else bool)
            for j, ex in enumerate(chunk):
                if args.span_labels:
                    gold[j, :len(ex['labels'])] = ex['labels']
                else:
                    gold[j, ex['positive_indices']] = True
            tags = np.zeros_like(gold, dtype=np.uint8)
            positive_counts = np.zeros(len(chunk), dtype=np.int64)
            cached_parts, cached_version = None, -1
            for i in range(int(lengths.max())):
                active = np.flatnonzero(i < lengths)
                epsilon = max(.1, 1. - .9 * total_steps / epsilon_decay)
                with torch.no_grad():
                    if cached_version != updates:
                        cached_parts = model.static_parts(words, distance)
                        cached_version = updates
                    q = model.from_static(cached_parts, torch.from_numpy(tags.astype(np.float32)).to(device))
                    decisions = q.argmax(dim=1).cpu().numpy()
                explore = np.random.random(len(chunk)) < epsilon
                decisions[explore] = np.random.randint(0, 3 if args.span_labels else 2, size=int(explore.sum()))
                before = tags.copy()
                tags[active, i] = decisions[active] + 1
                if args.span_labels and args.span_rate == 'mentions':
                    previous = before[:, i - 1].astype(np.int64) - 1 if i else np.zeros(len(chunk), dtype=np.int64)
                    positive_counts[active] += span_starts(decisions, previous)[active]
                else:
                    positive_counts[active] += decisions[active] != 0
                terminal = (i + 1 == lengths)
                rewards = reward_function(decisions, gold[:, i], positive_counts, features.num_words, terminal)
                for j in active:
                    transition = (chunk[j]['index'], before[j].copy(), int(decisions[j]), float(rewards[j]),
                                  tags[j].copy(), bool(terminal[j]))
                    if args.span_labels and args.span_replay == 'balanced':
                        transition += (int(gold[j, i]),)
                    replay.add(transition)
                    visited.add(chunk[j]['id'])
                    if terminal[j]:
                        epoch_visited.add(chunk[j]['id'])
                total_steps += len(active)
                vector_steps += 1
                if len(replay.rows) >= args.batch_size and vector_steps % args.train_frequency == 0:
                    batch = replay.sample(args.batch_size)
                    batch_examples = [features.all_examples[row[0]] for row in batch]
                    bw, bd = features.batch(batch_examples)
                    bbefore = torch.from_numpy(np.stack([row[1] for row in batch]).astype(np.float32)).to(device)
                    bafter = torch.from_numpy(np.stack([row[4] for row in batch]).astype(np.float32)).to(device)
                    actions = torch.tensor([row[2] for row in batch], device=device)
                    br = torch.tensor([row[3] for row in batch], device=device)
                    bt = torch.tensor([row[5] for row in batch], dtype=torch.bool, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = dqn_loss(model, target, bw, bbefore, bafter, bd, actions, br, bt)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite loss')
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise RuntimeError('Non-finite gradient')
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    updates += 1
                    if updates % args.target_updates == 0:
                        target.load_state_dict(model.state_dict())
                if time.time() - last_report >= 15:
                    progress = {'phase': 'training', 'epoch': epoch + 1, 'training_steps': total_steps,
                                'optimizer_updates': updates, 'epoch_examples_completed': len(epoch_visited),
                                'epoch_examples_total': len(train), 'epsilon': epsilon,
                                'recent_loss_mean': float(np.mean(losses[-100:])) if losses else None,
                                'elapsed_seconds': time.time() - start, 'best_dev_f1': best_f1}
                    write_json(args.output / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
                    last_report = time.time()
                if args.max_vector_steps and vector_steps >= args.max_vector_steps:
                    stop = True
                    break
            if stop:
                break
        epoch_complete = len(epoch_visited) == len(train)
        metrics, _ = evaluator(model, features, dev)
        probe, _ = evaluator(model, features, train[:min(32, len(train))])
        history.append({'epoch': epoch + 1, 'complete': epoch_complete, 'training_steps': total_steps,
                        'optimizer_updates': updates, 'mean_loss': float(np.mean(losses)) if losses else None,
                        'dev': metrics, 'train_probe': probe, 'seconds': time.time() - epoch_start})
        if metrics['f1'] > best_f1:
            best_f1, best_epoch, stale = metrics['f1'], epoch + 1, 0
            torch.save({'state_dict': model.state_dict(), 'configuration': configuration, 'epoch': epoch + 1}, args.output / 'best.pt')
        else:
            stale += 1
        resumable = {'state_dict': model.state_dict(), 'target_state': target.state_dict(),
                    'optimizer_state': optimizer.state_dict(), 'epoch': epoch + 1,
                    'training_steps': total_steps, 'configuration': configuration,
                    'vector_steps': vector_steps, 'updates': updates, 'best_f1': best_f1,
                    'best_epoch': best_epoch, 'stale': stale, 'history': history, 'visited': sorted(visited),
                    'replay': replay, 'initial_dev': initial_dev,
                    'feature_manifest_sha256': hashlib.sha256((args.features / 'manifest.json').read_bytes()).hexdigest(),
                    'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
        torch.save(resumable, args.output / 'last.pt.tmp')
        (args.output / 'last.pt.tmp').replace(args.output / 'last.pt')
        write_json(args.output / 'history.json', history)
        print('EPOCH_RESULT ' + json.dumps(history[-1]), flush=True)
        if stale >= args.patience or stop:
            break
    if not updates:
        raise RuntimeError('No optimizer updates')
    checkpoint = torch.load(args.output / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    test_metrics, predictions = evaluator(model, features, test)
    dev_metrics, _ = evaluator(model, features, dev)
    train_metrics, _ = evaluator(model, features, train)
    write_json(args.output / 'test_predictions.json', predictions)
    # Saving/reloading is checked on outputs from the same real test example.
    probe_words, probe_distance = features.batch(test[:1])
    probe_tags = torch.zeros((1, features.num_words), device=device)
    with torch.no_grad():
        first = model(probe_words, probe_tags, probe_distance)
        reloaded = model_class(args.mode).to(device)
        reloaded.load_state_dict(torch.load(args.output / 'best.pt', map_location=device, weights_only=False)['state_dict'])
        second = reloaded(probe_words, probe_tags, probe_distance)
        reload_equal = bool(torch.equal(first, second))
    changed = any(not torch.equal(initial[k], value.detach().cpu()) for k, value in model.state_dict().items())
    lexical = None
    if args.mode == 'act' and not args.span_labels:
        total, positive = collections.Counter(), collections.Counter()
        for ex in train:
            total.update(w.casefold() for w in ex['tokens'])
            positive.update(ex['tokens'][i].casefold() for i in ex['positive_indices'])
        chosen = [[i for i, w in enumerate(ex['tokens']) if total[w.casefold()] and positive[w.casefold()] / total[w.casefold()] >= .5] for ex in test]
        lexical, _ = scores(chosen, test)
    root = Path(__file__).resolve().parents[1]
    files = ['scripts/ceasdrl_torch.py', 'scripts/train_ceasdrl_risec.py', 'scripts/prepare_procedural_data.py', 'scripts/export_ceasdrl_features.py']
    if args.span_labels:
        files += ['scripts/ceasdrl_spans.py', 'scripts/risec_spans.py', 'scripts/prepare_ceasdrl_spans.py']
    receipt = {'status': 'completed' if changed and reload_equal else 'failed',
               'claim': 'Adapted cEASDRL {} extraction training, not original paper reproduction.'.format('BIO span' if args.span_labels else 'binary'),
               'configuration': configuration, 'initial_dev': initial_dev, 'best_epoch': best_epoch,
               'epochs_completed': sum(x['complete'] for x in history), 'history': history,
               'training_steps': total_steps, 'optimizer_updates': updates,
               'unique_train_examples_visited': len(visited),
               'examples': {s: len(x) for s, x in features.examples.items()},
               'dataset_sha256': features.manifest['dataset_sha256'],
               'feature_manifest_sha256': hashlib.sha256((args.features / 'manifest.json').read_bytes()).hexdigest(),
               'source_sha256': {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files},
               'parity': parity, 'skipped_examples': features.manifest['skipped'][args.mode],
               'train': train_metrics, 'dev': dev_metrics, 'test': test_metrics,
               'train_only_lexical_baseline_test': lexical,
               'weights_changed': changed, 'checkpoint_reload_predictions_equal': reload_equal,
               'losses_and_gradients_finite': True, 'total_seconds': time.time() - start,
               'limitations': ['Single seed; independent target network.',
                               'BIO boundary decisions extend the original binary output/reward; core binary parity is not a BIO benchmark reproduction.' if args.span_labels else 'First annotated token, not exact-span extraction.',
                               'Explicit argument links only; implicit links excluded.',
                               'Argument-stage test uses gold actions; report separate predicted-action pipeline results as the main metric.' if args.span_labels else 'Argument test uses gold actions and excludes annotated action heads.',
                               'No ES/OP/EX classification; no semantic-role classification.',
                               'Vectorized episodes, recorded hyperparameters and action window 600.',
                               'No NL2P comparison in this run.']}
    write_json(args.output / 'receipt.json', receipt)
    write_json(args.output / 'progress.json', {'phase': 'completed', 'best_epoch': best_epoch, 'test': test_metrics,
                                             'elapsed_seconds': time.time() - start})
    print('COMPLETED ' + json.dumps({'best_epoch': best_epoch, 'train': train_metrics, 'dev': dev_metrics, 'test': test_metrics,
                                    'seconds': receipt['total_seconds']}), flush=True)


if __name__ == '__main__':
    main()
