"""Offline cEASDRL feasibility training with the original Keras CNN and reward.

Run in Python 3.6 / TensorFlow 1.14 / Keras 2.3.1. This is a documented
adaptation, not a claim to reproduce published scores. No API requests.
"""
import argparse
import collections
import contextlib
import hashlib
import io
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'third_party' / 'contextual_drl'


def load_embeddings():
    from flair.embeddings import WordEmbeddings, ELMoEmbeddings, StackedEmbeddings
    cache = Path.home() / '.allennlp' / 'cache'
    files = {}
    for meta in cache.glob('*.json'):
        info = json.loads(meta.read_text(encoding='utf-8'))
        url = info.get('url', '')
        if '2x1024_128_2048cnn_1xhighway' in url:
            files['options' if 'options' in url else 'weights'] = str(meta.with_suffix(''))
    glove = Path.home() / '.flair' / 'embeddings' / 'glove.gensim'
    for path in [str(glove), files.get('options', ''), files.get('weights', '')]:
        if not path or not Path(path).is_file():
            raise FileNotFoundError('Missing local GloVe/ELMo cache: ' + path)
    return StackedEmbeddings([WordEmbeddings(str(glove)),
                             ELMoEmbeddings('small', options_file=files['options'], weight_file=files['weights'])])


def make_examples(records, mode, num_words):
    examples, skipped = [], []
    for row in records:
        if len(row['words']) > num_words and mode == 'act':
            raise ValueError('Document exceeds action window: ' + row['id'])
        if mode == 'act':
            examples.append({'id': row['id'], 'doc_id': row['id'], 'tokens': row['words'],
                             'positive_indices': [a['trigger_tokens'][0] for a in row['actions']],
                             'act2related': {}})
        else:
            if not row['argument_links_available']:
                raise ValueError('No annotated action-argument links in ' + row['source'])
            heads = [a['trigger_tokens'][0] for a in row['actions']]
            for i, act in enumerate(row['actions']):
                head = act['trigger_tokens'][0]
                # Same 100-token capacity as upstream; deterministic centered window.
                start = max(0, min(head - num_words // 2, len(row['words']) - num_words))
                stop = min(len(row['words']), start + num_words)
                args = [a['token_indices'][0] for a in act['explicit_arguments']]
                if any(not start <= j < stop for j in args):
                    skipped.append({'doc_id': row['id'], 'action': i, 'reason': 'argument_outside_window'})
                    continue
                examples.append({'id': '{}:{}'.format(row['id'], i), 'doc_id': row['id'],
                                 'tokens': row['words'][start:stop],
                                 'positive_indices': sorted(set(j - start for j in args)),
                                 'act_ind': head - start, 'act_inds': [h - start for h in heads if start <= h < stop],
                                 'obj_inds': [sorted(set(j - start for j in args)), []],
                                 'implicit_arguments_excluded': len(act['implicit_arguments'])})
    return examples, skipped


def args_for(mode, num_words, dim):
    return SimpleNamespace(word_dim=dim, tag_dim=dim, dis_dim=dim, dropout=.5,
                           optimizer='adam', dense_dim=256, batch_size=4, gamma=.9,
                           learning_rate=.001, num_actions=2, num_filters=32,
                           num_words=num_words, context_len=num_words, contextual_embedding='elmo')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['act', 'arg'], default='act')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=128, help='0 means full epochs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--target', choices=['independent', 'upstream-shared'], default='independent')
    parser.add_argument('--check-pretrained', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.max_steps < 0 or args.threads < 1:
        parser.error('epochs/threads must be positive and max-steps must be nonnegative')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Refusing to overwrite an existing run: ' + str(args.output))
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    start_time = time.time()
    sys.path.insert(0, str(VENDOR))
    import numpy as np
    import tensorflow as tf
    import keras
    import torch
    from keras.backend.tensorflow_backend import set_session
    from keras.models import clone_model
    from flair.data import Sentence, Token
    from EADQN import DeepQLearner
    from cEnvironment import Environment
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    set_session(tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=args.threads,
                                                inter_op_parallelism_threads=1)))
    print('Loading cached GloVe + ELMo-small', flush=True)
    embeddings = load_embeddings()
    dim = embeddings.embedding_length
    if dim != 868:
        raise ValueError('Expected original ELMo variant dimension 868, got {}'.format(dim))
    num_words = 500 if args.mode == 'act' else 100
    cfg = args_for(args.mode, num_words, dim)
    records, examples, skipped = {}, {}, {}
    embedding_cache = {}
    for split in ['train', 'dev', 'test']:
        records[split] = [json.loads(line) for line in (args.data / (split + '.jsonl')).read_text(encoding='utf-8').splitlines() if line.strip()]
        examples[split], skipped[split] = make_examples(records[split], args.mode, num_words)
        if not examples[split]:
            raise ValueError('Empty split ' + split)
        for ex in examples[split]:
            key = tuple(ex['tokens'])
            if key not in embedding_cache:
                sent = Sentence()
                for word in key:
                    sent.add_token(Token(word))
                with torch.no_grad():
                    embeddings.embed(sent)
                vec = np.stack([t.embedding.detach().cpu().numpy() for t in sent])
                if vec.shape != (len(key), dim):
                    raise ValueError('Embedding/token alignment mismatch')
                embedding_cache[key] = vec
            vec = np.zeros((num_words, dim), dtype=np.float32)
            vec[:len(key)] = embedding_cache[key]
            if args.mode == 'arg':
                distance = np.zeros((num_words, dim), dtype=np.float32)
                distance[:len(key)] = np.abs(np.arange(len(key)) - ex['act_ind'])[:, None]
                vec = np.concatenate([vec, distance], axis=1)
            ex['features'] = vec
        print('Embedded {}: {} documents / {} examples'.format(split, len(records[split]), len(examples[split])), flush=True)
    sets = [set(r['id'] for r in records[s]) for s in ['train', 'dev', 'test']]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError('Document leakage')
    with contextlib.redirect_stdout(io.StringIO()):
        net = DeepQLearner(cfg, args.mode, 'channels_last')
    upstream_shared = net.model.weights[0] is net.target_model.weights[0]
    if args.target == 'independent':
        net.target_model = clone_model(net.model)
        net.update_target_network()
    pretrained = None
    initial_weights = net.model.get_weights()
    probe_ex = examples['train'][0]
    probe = np.concatenate([probe_ex['features'], np.zeros((num_words, 1))], axis=1)
    if args.check_pretrained:
        path = ROOT / 'data' / 'ceasdrl' / 'weights' / ('cooking_{}_elmo.h5'.format(args.mode))
        net.load_weights(str(path))
        q = net.predict(probe)
        pretrained = {'path': str(path), 'q_shape': list(q.shape), 'finite': bool(np.isfinite(q).all()),
                      'purpose': 'Weight/architecture compatibility only; not a benchmark score.'}
        net.model.set_weights(initial_weights)
        net.update_target_network()
    env = Environment.__new__(Environment)
    env.agent_mode, env.num_words = args.mode, num_words
    env.ra, env.reward_base, env.action_rate, env.use_act_rate = [1, 2, 3], 50., .10, 1

    def reset(ex):
        env.current_text = ex
        gold = np.ones((num_words, 1), dtype=np.float32)
        gold[ex['positive_indices']] = 2
        env.text_vec = np.concatenate([ex['features'], gold], axis=1)
        env.state = np.concatenate([ex['features'], np.zeros((num_words, 1))], axis=1)
        env.terminal_flag = False

    def evaluate(items, predictor=None):
        right = predicted = truth = 0
        outputs = []
        for ex in items:
            reset(ex)
            chosen = []
            for i in range(len(ex['tokens'])):
                action = predictor(ex, i) if predictor else int(np.argmax(net.predict(env.state)[0]))
                env.act(action, i)
                if action:
                    chosen.append(i)
            # Apply the same upstream exclusion of action heads for argument scoring.
            if args.mode == 'arg':
                chosen = [i for i in chosen if i not in ex['act_inds']]
            gold = set(ex['positive_indices'])
            right += len(set(chosen) & gold)
            predicted += len(chosen)
            truth += len(gold)
            outputs.append({'id': ex['id'], 'predicted_indices': chosen, 'gold_indices': sorted(gold)})
        precision = right / predicted if predicted else 0.
        recall = right / truth if truth else 0.
        return {'precision': precision, 'recall': recall, 'f1': 2 * right / (predicted + truth) if predicted + truth else 0.,
                'right': right, 'predicted': predicted, 'gold': truth}, outputs

    # Store document references + compact tag vectors, not 50,000 copies of embeddings.
    replay, losses, history = [], [], []
    global_step, best_f1, best_weights = 0, -1., None
    training_start = time.time()
    for epoch in range(args.epochs):
        order = list(examples['train'])
        random.shuffle(order)
        for ex in order:
            reset(ex)
            for i in range(len(ex['tokens'])):
                eps = max(.1, 1 - .9 * global_step / 1000.)
                action = random.randrange(2) if random.random() < eps else int(np.argmax(net.predict(env.state)[0]))
                before = env.state[:, -1].astype(np.uint8).copy()
                reward = env.act(action, i)
                terminal = env.isTerminal()
                if terminal:
                    reward += 2
                replay.append((ex, before, action, reward, env.state[:, -1].astype(np.uint8).copy(), terminal))
                if len(replay) > 50000:
                    replay.pop(0)
                if global_step % 5 == 0:
                    net.update_target_network()
                if len(replay) > cfg.batch_size:
                    # Reproduce the positive/negative reward sampling preference.
                    pos = [x for x in replay if x[3] > 0]
                    neg = [x for x in replay if x[3] <= 0]
                    batch = [random.choice(pos or replay) for _ in range(int(.9 * cfg.batch_size))]
                    batch += [random.choice(neg or replay) for _ in range(cfg.batch_size - len(batch))]
                    pre = np.stack([np.column_stack([x[0]['features'], x[1]]) for x in batch])
                    post = np.stack([np.column_stack([x[0]['features'], x[4]]) for x in batch])
                    loss = float(net.train((pre, np.array([x[2] for x in batch]), np.array([x[3] for x in batch]), post, np.array([x[5] for x in batch]))))
                    if not np.isfinite(loss):
                        raise RuntimeError('Non-finite loss')
                    losses.append(loss)
                global_step += 1
                if global_step % 32 == 0:
                    print('Training step {} / loss {:.4f}'.format(global_step, losses[-1] if losses else 0.), flush=True)
                if args.max_steps and global_step >= args.max_steps:
                    break
            if args.max_steps and global_step >= args.max_steps:
                break
        metrics, _ = evaluate(examples['dev'])
        history.append({'epoch': epoch, 'step': global_step, 'dev': metrics})
        if metrics['f1'] > best_f1:
            best_f1, best_weights = metrics['f1'], net.model.get_weights()
        if args.max_steps and global_step >= args.max_steps:
            break
    if not losses:
        raise RuntimeError('No optimizer update was performed')
    net.model.set_weights(best_weights)
    test, predictions = evaluate(examples['test'])
    dev, _ = evaluate(examples['dev'])
    lexical = None
    if args.mode == 'act':
        total_counts, positive_counts = collections.Counter(), collections.Counter()
        for ex in examples['train']:
            total_counts.update(w.casefold() for w in ex['tokens'])
            positive_counts.update(ex['tokens'][i].casefold() for i in ex['positive_indices'])
        def lex_predict(ex, i):
            w = ex['tokens'][i].casefold()
            return int(total_counts[w] > 0 and positive_counts[w] / total_counts[w] >= .5)
        lexical, _ = evaluate(examples['test'], lex_predict)
    changed = any(not np.array_equal(a, b) for a, b in zip(initial_weights, best_weights))
    path = args.output / 'best.weights.h5'
    net.save_weights(str(path))
    q_before = net.predict(probe)
    net.model.set_weights(initial_weights)
    net.load_weights(str(path))
    reload_ok = bool(np.allclose(q_before, net.predict(probe)))
    receipt = {'status': 'smoke_pass' if changed and reload_ok else 'failed',
               'claim': 'Feasibility pilot with a fixed training budget; not paper reproduction or a converged-model comparison.',
               'mode': args.mode, 'embedding': 'GloVe-100 + ELMo-small-768', 'embedding_dim': dim,
               'seed': args.seed, 'target_mode': args.target, 'upstream_target_shared_weights_observed': upstream_shared,
               'python': sys.version, 'tensorflow': tf.__version__, 'keras': keras.__version__, 'torch': torch.__version__,
               'device': 'CPU', 'num_words': num_words, 'batch_size': cfg.batch_size,
               'dataset_documents': {s: [r['id'] for r in records[s]] for s in records},
               'dataset_sha256': {s: hashlib.sha256((args.data / (s + '.jsonl')).read_bytes()).hexdigest() for s in records},
               'examples': {s: len(v) for s, v in examples.items()}, 'skipped_examples': skipped,
               'training_steps': global_step, 'optimizer_updates': len(losses),
               'loss_first': losses[0], 'loss_last': losses[-1], 'losses_finite': True,
               'weights_changed': changed, 'checkpoint_reload_predictions_equal': reload_ok,
               'pretrained_compatibility': pretrained, 'dev': dev, 'test': test, 'history': history,
               'train_only_lexical_baseline_test': lexical,
               'dataset_sources': sorted(set(r['source'] for values in records.values() for r in values)),
               'annotation_scopes': sorted(set(r.get('annotation_scope', 'Annotated action spans') for values in records.values() for r in values)),
               'argument_evaluation': 'Gold-action-conditioned explicit first-token arguments only; not end-to-end.' if args.mode == 'arg' else None,
               'train_and_eval_seconds': time.time() - training_start,
               'total_seconds': time.time() - start_time,
               'deviations': ['Independent target network by default', 'Document-disjoint splits (RISeC official; others custom)',
                              'Compact replay with explicit pre/post states', 'Batch size 4 for local smoke',
                              'Whole-example contextual embedding', 'First-token span projection',
                              'Centered argument context; implicit arguments excluded and recorded',
                              'No inferred ES/OP/EX labels; every annotated occurrence is an extraction target']}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    (args.output / 'test_predictions.json').write_text(json.dumps(predictions, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
