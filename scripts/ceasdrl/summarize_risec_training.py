"""Summarize completed RISeC runs, independently recount predictions, plot curves."""
import csv
import hashlib
import json
from pathlib import Path

try:
    from .source_receipts import matches_source
except ImportError:
    from source_receipts import matches_source


def main():
    root = Path(__file__).resolve().parents[2]
    runs = root / 'results/ceasdrl_risec_20261118'
    summary = {'dataset': 'RISeC', 'split_documents': {'train': 126, 'dev': 56, 'test': 78},
               'seed': 0, 'es_ex_op': False, 'runs': {}, 'integrity_checks': {}}
    table = []
    for mode in ['act', 'arg']:
        directory = runs / (mode + '_seed0')
        receipt = json.loads((directory / 'receipt.json').read_text(encoding='utf-8'))
        predictions = json.loads((directory / 'test_predictions.json').read_text(encoding='utf-8'))
        right = sum(len(set(x['predicted_indices']) & set(x['gold_indices'])) for x in predictions)
        predicted = sum(len(set(x['predicted_indices'])) for x in predictions)
        gold = sum(len(set(x['gold_indices'])) for x in predictions)
        assert (right, predicted, gold) == tuple(receipt['test'][k] for k in ['right', 'predicted', 'gold'])
        assert receipt['status'] == 'completed'
        assert receipt['weights_changed'] and receipt['checkpoint_reload_predictions_equal']
        assert receipt['losses_and_gradients_finite']
        for split, expected in receipt['dataset_sha256'].items():
            path = root / 'data/procedural/risec/converted' / (split + '.jsonl')
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        for relative, expected in receipt['source_sha256'].items():
            assert matches_source(relative, expected), relative
        summary['runs'][mode] = {k: receipt[k] for k in ['configuration', 'best_epoch', 'epochs_completed',
                                                       'training_steps', 'optimizer_updates', 'examples',
                                                       'train', 'dev', 'test', 'total_seconds']}
        summary['integrity_checks'][mode] = 'Dataset/source hashes and independently recounted test predictions match'
        table.append({'task': mode, **receipt['test']})
        if mode == 'act':
            summary['lexical_action_baseline'] = receipt['train_only_lexical_baseline_test']
            table.append({'task': 'lexical_act', **receipt['train_only_lexical_baseline_test']})
    pipeline = json.loads((runs / 'pipeline_seed0/receipt.json').read_text(encoding='utf-8'))
    assert pipeline['uses_predicted_actions']
    # Rebuild pair sets independently from human-readable pipeline outputs.
    predictions = json.loads((runs / 'pipeline_seed0/predictions.json').read_text(encoding='utf-8'))
    predicted_pairs = {(p['doc_id'], p['action_index'], a) for p in predictions for a in p['argument_indices']}
    records = [json.loads(x) for x in (root / 'data/procedural/risec/converted/test.jsonl').read_text(encoding='utf-8').splitlines()]
    gold_pairs = {(r['id'], a['trigger_tokens'][0], arg['token_indices'][0])
                  for r in records for a in r['actions'] for arg in a['explicit_arguments']}
    counts = (len(predicted_pairs & gold_pairs), len(predicted_pairs), len(gold_pairs))
    assert counts == tuple(pipeline['action_argument_pair'][k] for k in ['right', 'predicted', 'gold'])
    summary['pipeline'] = {k: pipeline[k] for k in ['action_argument_pair', 'exact_action_argument_set', 'predicted_actions', 'gold_actions']}
    summary['integrity_checks']['pipeline'] = 'Pair counts independently rebuilt; missed actions remain in gold denominator'
    table.append({'task': 'pipeline_pairs', **pipeline['action_argument_pair']})
    table.append({'task': 'pipeline_exact_frames', **pipeline['exact_action_argument_set']})
    (runs / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (runs / 'metrics.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['task', 'precision', 'recall', 'f1', 'right', 'predicted', 'gold'])
        writer.writeheader()
        writer.writerows(table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for mode, label, color in [('act', 'Action', '#146c94'), ('arg', 'Explicit argument (gold action)', '#be5a38')]:
        history = json.loads((runs / (mode + '_seed0') / 'history.json').read_text())
        epochs = [x['epoch'] for x in history]
        axes[0].plot(epochs, [100 * x['dev']['f1'] for x in history], marker='o', color=color, label=label)
        axes[1].plot(epochs, [x['mean_loss'] for x in history], marker='o', color=color, label=label)
    axes[0].set(ylabel='Validation first-token F1 (%)', xlabel='Completed epoch', ylim=(0, 100))
    axes[1].set(ylabel='Mean DQN loss', xlabel='Completed epoch', yscale='log')
    for ax in axes:
        ax.set_xticks(range(1, 6))
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('RISeC / adapted cEASDRL / seed 0 / no ES-EX-OP labels')
    fig.savefig(runs / 'training_curves.png', dpi=160)
    plt.close(fig)
    print(json.dumps({'metrics': table, 'integrity_checks': summary['integrity_checks']}, indent=2))


if __name__ == '__main__':
    main()
