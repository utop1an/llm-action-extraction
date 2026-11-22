"""Inspect the released OpenPI-A labels without silently adjudicating them."""
import collections
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / 'data/procedural/openpi_a'
    silver = [json.loads(x) for x in (root / 'data/silver_standard.jsonl').read_text().splitlines() if x.strip()]
    units = [json.loads(x) for x in (root / 'data/validation/validation_input.jsonl').read_text().splitlines() if x.strip()]
    annotators = {name: json.loads((root / 'data/validation' / ('annotator_' + name + '.json')).read_text())['annotations'] for name in ['A', 'B']}
    common = set(annotators['A']) & set(annotators['B'])
    both_retained = [key for key in common if all(annotators[n][key]['action'] != 'trash' for n in annotators)]
    same = sum(annotators['A'][key].get('selected_span') == annotators['B'][key].get('selected_span') for key in both_retained)
    result = {'silver_rows': len(silver), 'silver_source_documents': len(set(x['doc_id'] for x in silver)),
              'silver_unique_steps': len(set((x['doc_id'], x['step_id']) for x in silver)),
              'topics': dict(collections.Counter(x['topic'] for x in silver)),
              'validation_units': len(units),
              'annotator_actions': {name: dict(collections.Counter(x['action'] for x in values.values())) for name, values in annotators.items()},
              'changed_trigger_among_nontrashed': {name: sum(bool(x.get('changed')) for x in values.values() if x['action'] != 'trash') for name, values in annotators.items()},
              'common_units': len(common), 'both_not_trashed': len(both_retained),
              'identical_selected_span_objects_among_retained': same,
              'scope': 'Silver file is one selected trigger per step; validation units associate entities and state changes. Neither supplies all planning-action occurrences.',
              'adjudication': 'None. Original responses retained. Object equality here is an audit count, not a reproduction of the paper agreement metric.'}
    (root / 'release_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
