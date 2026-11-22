"""Create a verified, compact report from completed full-name experiments."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from .source_receipts import matches_source
except ImportError:
    from source_receipts import matches_source
import xml.etree.ElementTree as ET


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--select-action', action='store_true', help='Freeze the action candidate using dev scores only, before pipeline testing')
    p.add_argument('--action-runs', nargs='+', default=['act_seed0', 'act_lr3e4_seed0', 'act_balanced_seed0'])
    p.add_argument('--argument-run', default='arg_balanced_seed0')
    args = p.parse_args()
    root = args.run_root
    repository = Path(__file__).resolve().parents[2]
    if args.select_action:
        if (root / 'selection.json').exists():
            raise ValueError('Action selection is already frozen')
        candidates = []
        for name in args.action_runs:
            receipt = read(root / name / 'receipt.json')
            if receipt['status'] != 'completed' or receipt['configuration']['mode'] != 'act':
                raise ValueError('Incomplete or invalid action candidate: ' + name)
            candidates.append({'run': name, 'dev_f1': receipt['dev']['f1'], 'best_epoch': receipt['best_epoch'],
                               'epochs_completed': receipt['epochs_completed'], 'configuration': receipt['configuration']})
        chosen = max(candidates, key=lambda c: c['dev_f1'])
        selection = {'action_run': chosen['run'], 'argument_run': args.argument_run, 'action_candidates': candidates,
                     'criterion': 'Highest dev exact-span micro F1; ties retain the first listed candidate.',
                     'test_scores_used_for_selection': False, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
                     'action_checkpoint_sha256': sha(root / chosen['run'] / 'best.pt')}
        (root / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
        print(json.dumps({'selected': chosen['run'], 'dev_f1': chosen['dev_f1']}))
        return
    selection = read(root / 'selection.json')
    action_path, argument_path = root / selection['action_run'], root / selection['argument_run']
    stages = {'act': read(action_path / 'receipt.json'), 'arg': read(argument_path / 'receipt.json')}
    pipeline = read(root / 'pipeline_seed0/receipt.json')
    old = read(root / 'legacy_name_recheck.json')
    selected_dev = stages['act']['dev']['f1']
    assert selected_dev == max(c['dev_f1'] for c in selection['action_candidates'])
    checks = []
    snapshots = {sha(p): p for p in (root / 'source_history').glob('*.py')}
    for mode, stage in stages.items():
        assert stage['status'] == 'completed'
        assert stage['weights_changed'] and stage['checkpoint_reload_predictions_equal']
        assert stage['losses_and_gradients_finite']
        assert stage['best_epoch'] == pipeline['best_epochs'][mode]
        assert stage['dev']['f1'] == max(h['dev']['f1'] for h in stage['history'])
        for filename, expected in stage['source_sha256'].items():
            current = Path(__file__).with_name(Path(filename).name)
            matched = snapshots.get(expected)
            if matched is None and matches_source(filename, expected):
                matched = current
            if matched is None:
                raise ValueError('Unverifiable training source: ' + filename)
            checks.append({'stage': mode, 'source': filename, 'matching_file': str(matched), 'sha256': expected,
                           'matching_file_sha256': sha(matched)})
        run = action_path if mode == 'act' else argument_path
        assert sha(run / 'best.pt') == pipeline['checkpoint_sha256'][mode]
        compatibility = read(run / 'trained_keras_compatibility.json')
        assert compatibility['passed'] and compatibility['num_q_outputs'] == 3
    assert pipeline['status'] == 'completed'
    assert not pipeline['inference_uses_gold_labels'] and not pipeline['gold_name_expansion']
    assert pipeline['metrics']['action_name']['gold'] == 880
    assert pipeline['metrics']['linked_argument_name']['gold'] == 1616
    for split, expected in stages['act']['dataset_sha256'].items():
        assert stages['arg']['dataset_sha256'][split] == expected
        assert sha(repository / 'data/procedural/risec/converted' / (split + '.jsonl')) == expected
    validation = ET.parse(root / 'validation.xml').getroot()
    suites = [validation] if validation.tag == 'testsuite' else list(validation)
    assert all(int(s.attrib.get('failures', 0)) == 0 and int(s.attrib.get('errors', 0)) == 0 for s in suites)
    tests = sum(int(s.attrib['tests']) for s in suites)
    smoke = read(root / 'plain_text_smoke/verification.json')
    assert smoke['passed'] and smoke['batch_and_plain_text_outputs_identical']
    assert smoke['checkpoint_sha256'] == pipeline['checkpoint_sha256']
    summary = {'status': 'verified', 'metrics': pipeline['metrics'], 'previous_metrics_same_evaluator': old['metrics'],
               'selection': selection, 'tests_passed': tests, 'training_source_checks': checks, 'plain_text_inference_check': smoke,
               'stages': {m: {'best_epoch': s['best_epoch'], 'epochs_completed': s['epochs_completed'],
                              'dev_f1': s['dev']['f1'], 'training_steps': s['training_steps'],
                              'optimizer_updates': s['optimizer_updates']} for m, s in stages.items()},
               'limitations': ['Adapted CNN/DQN with BIO; not original paper reproduction.',
                               'One random seed, official 126/56/78 document split.',
                               'Frozen legacy relation policy: non-action, non-condition targets, including 45 TODO relations (10 in test); roles are not predicted.',
                               'The legacy implicit_arguments field also contains temporal and other action-to-action relations; 1695 is not a pure implicit-argument count.',
                               'Candidate selection reads dev only, but the training harness precomputes candidate test scores before cross-candidate selection; not a fully sealed holdout.',
                               'Earlier first-token argument training only completed three epochs; before/after is not a budget-controlled ablation.',
                               'No LLM/NL2P comparison was run.']}
    audit_path = root / 'audit_review/final/audit.json'
    audit = read(audit_path) if audit_path.exists() else None
    if audit is not None:
        assert audit['status'] == 'passed_with_scope_findings'
        assert audit['inference']['fresh_all_document_predictions_identical']
        for mode, name in [('act', selection['action_run']), ('arg', selection['argument_run'])]:
            assert audit['artifact_sha256'][name + '/best.pt'] == pipeline['checkpoint_sha256'][mode]
        for filename, expected in audit['artifact_sha256'].items():
            assert sha(root / filename) == expected
        summary['status'] = 'verified_with_scope_findings'
        summary['independent_audit'] = {'path': str(audit_path), 'sha256': sha(audit_path),
                                        'metrics_by_reference_policy': audit['independent_character_span_metrics'],
                                        'split_counts': audit['split_counts'],
                                        'no_retraining': True}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (root / 'learning_curves.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['run', 'epoch', 'complete', 'dev_exact_span_f1', 'train_probe_f1', 'mean_dqn_loss', 'optimizer_updates'])
        for name in [c['run'] for c in selection['action_candidates']] + [selection['argument_run']]:
            for h in read(root / name / 'history.json'):
                writer.writerow([name, h['epoch'], h['complete'], h['dev']['f1'], h['train_probe']['f1'], h['mean_loss'], h['optimizer_updates']])
    lines = ['# RISeC 完整动作与参数名称：修复后的实测结果', '',
             '输入完整文本，先预测动作，再预测对应参数。以下参数指标使用模型预测的动作，包含动作漏检造成的参数漏检。', '',
             '| 测试指标 | 旧版 F1 | 修复后 Precision | 修复后 Recall | 修复后 F1 |',
             '|---|---:|---:|---:|---:|']
    for key, label in [('action_name', '完整动作名称'), ('linked_argument_name', '完整参数名称及其所属动作')]:
        m = pipeline['metrics'][key]
        lines.append('| {} | {:.2f}% | {:.2f}% | {:.2f}% | **{:.2f}%** |'.format(label, old['metrics'][key]['f1'] * 100,
                     m['precision'] * 100, m['recall'] * 100, m['f1'] * 100))
    lines += ['', '官方文档划分为 **126 train / 56 dev / 78 test**。测试集有 880 个动作和 1,616 条纳入评测的显式参数关系。', '',
              '名称按完整标注短语匹配，仅规范化大小写和空白；还需匹配出现位置。参数必须属于正确的动作。没有使用金标准补全预测名称。', '',
              '这是 **cEASDRL 的适配版**：保留 CNN/DQN，增加 BIO 边界标签；加强候选按短语起点计算奖励中的提取数量，并平衡 O/B/I 的正奖励回放。它不是原论文指标复现。', '',
              '未训练或评测 ES/EX/OP、语义角色类别、隐含参数及条件从句参数。当前只有一个随机种子。旧参数模型只完成了三轮，因此前后对比同时包含标签修复和训练变化，不能视为单因素消融。', '',
              '## 选模与检查', '', '| 阶段 | 运行目录 | 完成轮数 | 最佳轮数 | 验证集完整短语 F1 |', '|---|---|---:|---:|---:|']
    for mode, name in [('act', selection['action_run']), ('arg', selection['argument_run'])]:
        s = stages[mode]
        lines.append('| {} | `{}` | {} | {} | {:.2f}% |'.format(mode, name, s['epochs_completed'], s['best_epoch'], s['dev']['f1'] * 100))
    lines += ['', '模型仅按验证集选取。{} 项测试通过；两个训练检查点均完成保存/重载检查，并已在原 Keras 网络设置三个输出后核对数值。数据、模型和训练源码哈希已复核。'.format(tests), '',
              '## 文件', '', '- `pipeline_seed0/predictions.json`：78 篇全文、预测动作、对应参数及字符位置。',
              '- `pipeline_seed0/errors.json`：动作和关联参数的 FP/FN。',
              '- `pipeline_seed0/receipt.json`：端到端评测计数与模型哈希。',
              '- `selection.json`：所有动作候选的验证集分数和最终选择。',
              '- `learning_curves.csv`：训练损失和验证集完整短语 F1。',
              '- `summary.json`：汇总及检查记录。', '',
              '纯文本预测入口：`scripts/ceasdrl/predict_ceasdrl_text.py`。完整方法与命令见 `docs/dev-notes/2026-11-18-risec-spans-v2.md`。', '']
    lines += ['', '无标注纯文本预测与批量评测输出完全一致。`prediction_example.md` 保存了一篇完整原文及未经人工修正的模型输出。', '']
    if audit is not None:
        clean_f1 = audit['independent_character_span_metrics']['defined_noncondition']['linked_argument']['f1'] * 100
        lines.insert(2, '**复核补充：下表沿用冻结的原转换口径，参数测试目标包含 10 条 `TODO` 关系。保持模型和预测不变，仅排除这些参考关系，参数 F1 为 {:.2f}%。这不是清理数据后重新训练的结果。完整说明见 [复核报告](audit_review/AUDIT.md)。**\n'.format(clean_f1))
        lines += ['', '## 复核发现与限制', '',
                  '- 旧转换中的 `implicit_arguments` 是混合排除集合，1,695 条不能全部称为隐含参数；其中也有动作间时序等关系。',
                  '- 参数范围包含食材、工具、时间、温度、其他修饰语和目的从句，但排除条件从句及动作目标关系；不是 PDDL 变量或完整动作模型。',
                  '- 候选选模代码仅依据 dev，但训练脚本提前生成了候选 test 分数，因此不能称为测试集全程封存。',
                  '- 本次重新运行 78 篇去标注输入，预测全部一致；独立使用原始 BRAT 字符位置重算分数，未复用项目计分函数。',
                  '- 本次是核验和报告纠正，没有重训、改变模型预测或人工修正所有原始标注。', '']
    (root / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'metrics': pipeline['metrics'], 'tests': tests}, indent=2))


if __name__ == '__main__':
    main()
