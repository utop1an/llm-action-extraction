"""Lossless published RiCoRecA ACTION/explicit-member projection for cEASDRL.

Published memberships identify action names; original Prodigy graph paths
identify their occurrences. No closest-verb inference or text imputation.
"""
import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

try:
    from .prepare_procedural_data import tokenize, span_tokens
    from .prepare_ricoreca_split import make_split, sha
    from .risec_spans import make_span_examples
except ImportError:
    from prepare_procedural_data import tokenize, span_tokens
    from prepare_ricoreca_split import make_split, sha
    from risec_spans import make_span_examples

PREDICATES = {'ACTION', 'IF', 'UNTIL', 'REPEAT'}


def published_records(raw_root):
    """Use the released CSV union as corpus membership, not its old folds."""
    rows, sources = {}, []
    files = sorted(Path(raw_root).rglob('Cross_Val/*.csv'))
    if len(files) != 15:
        raise ValueError('Expected 15 released cross-validation CSVs')
    for path in files:
        count = 0
        for row in csv.DictReader(path.open(encoding='utf-8-sig', newline='')):
            item = {k: row[k] for k in ('id', 'recipe', 'out')}
            if item['id'] in rows and rows[item['id']] != item:
                raise ValueError('Conflicting published annotation versions')
            rows[item['id']] = item
            count += 1
        sources.append({'path': str(path.relative_to(raw_root)), 'sha256': sha(path), 'records': count})
    return rows, sources


def parse_published(row):
    parts, spans, pos, previous = [], [], 0, 0
    for match in re.finditer(r'\[([^\[\]]*)\]', row['out']):
        plain = row['out'][previous:match.start()]
        parts.append(plain)
        pos += len(plain)
        fields = [x.strip() for x in match.group(1).split('|')]
        if len(fields) != 5 or not fields[0]:
            raise ValueError('Invalid published annotation')
        spans.append({'start': pos, 'end': pos + len(fields[0]), 'text': fields[0],
                      'antecedents': fields[1], 'memberships': fields[2].split(', '),
                      'label': fields[3], 'role_links': fields[4]})
        pos += len(fields[0])
        parts.append(fields[0])
        previous = match.end()
    parts.append(row['out'][previous:])
    if ''.join(parts) != row['recipe']:
        raise ValueError('Published annotations do not reconstruct exact source text')
    return spans


def graph_memberships(raw, start):
    """Stop at first predicates on each graph path, then apply Join sharing.

    Member/Or/Modifier traversal follows the author's recursive membership
    projection. Dependency is never traversed; predicates are barriers.
    Returns auditable paths using raw relation indices and token-end IDs.
    """
    nodes = {s['token_end']: s for s in raw['spans']}
    if len(nodes) != len(raw['spans']):
        raise ValueError('Ambiguous raw span IDs')
    graph, joins = collections.defaultdict(list), collections.defaultdict(list)
    for index, rel in enumerate(raw['relations']):
        h, c, label = rel['head'], rel['child'], rel['label']
        if h not in nodes or c not in nodes:
            # The release contains one obsolete Member edge to an unlabeled
            # token. Keep it in the audit; never use it to fabricate a path.
            # Any affected published membership still fails resolution below.
            continue
        target = graph if label in ('Member', 'Or', 'Modifier') else joins if label == 'Join' else None
        if target is not None:
            target[h].append((c, index, label))
            target[c].append((h, index, label))
    queue, seen, reached = collections.deque([(start, [])]), set(), {}
    while queue:
        node, path = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        if nodes[node]['label'].upper() in PREDICATES:
            reached[node] = path
            continue
        for nxt, index, label in sorted(graph[node]):
            queue.append((nxt, path + [{'relation_index': index, 'label': label, 'from': node, 'to': nxt}]))
    queue = collections.deque(sorted(reached))
    while queue:
        node = queue.popleft()
        for nxt, index, label in sorted(joins[node]):
            if nodes[nxt]['label'].upper() not in PREDICATES:
                raise ValueError('Join must connect predicates')
            if nxt not in reached:
                reached[nxt] = reached[node] + [{'relation_index': index, 'label': label, 'from': node, 'to': nxt}]
                queue.append(nxt)
    return reached


def convert_record(raw, released):
    if raw['text'] != released['recipe'] or raw['meta']['id'] != released['id']:
        raise ValueError('Raw/published identity mismatch')
    tokens = tokenize(raw['text'])
    bychar = {(s['start'], s['end']): s for s in raw['spans']}
    spans = parse_published(released)
    for span in spans:
        source = bychar.get((span['start'], span['end']))
        if source is None or source['label'].upper() != span['label']:
            raise ValueError('Published/raw span mismatch')
        span['source_id'] = source['token_end']
        span['token_indices'] = span_tokens(tokens, [(span['start'], span['end'])])
    actions = {s['source_id']: {'source_id': s['source_id'], 'trigger_tokens': s['token_indices'],
                              'char_span': [s['start'], s['end']], 'text': s['text'],
                              'explicit_arguments': [], 'implicit_arguments': []}
               for s in spans if s['label'] == 'ACTION'}
    names = {a['text'] for a in actions.values()}
    excluded = []
    for span in spans:
        if span['label'] in PREDICATES:
            continue
        wanted = set(span['memberships']) & names
        if not wanted:
            excluded.append({'source_id': span['source_id'], 'text': span['text'],
                             'memberships': span['memberships'], 'reason': 'Member of control predicate, not ACTION'})
            continue
        reached = graph_memberships(raw, span['source_id'])
        found = {i: path for i, path in reached.items() if i in actions and actions[i]['text'] in wanted}
        if {actions[i]['text'] for i in found} != wanted or len(found) != len(wanted):
            raise ValueError('Unresolved or ambiguous action occurrence: {} {}'.format(released['id'], span))
        for action_id, path in sorted(found.items()):
            actions[action_id]['explicit_arguments'].append({
                'target_text': span['text'], 'token_indices': span['token_indices'],
                'char_span': [span['start'], span['end']], 'source_id': span['source_id'],
                'source_type': span['label'], 'published_memberships': span['memberships'],
                'relation_path': path})
    record = {'id': released['id'], 'source': 'RiCoRecA', 'text': raw['text'],
              'words': [t[0] for t in tokens], 'token_offsets': [[t[1], t[2]] for t in tokens],
              'argument_links_available': True, 'actions': sorted(actions.values(), key=lambda a: a['trigger_tokens'][0]),
              'excluded_control_members': excluded,
              'invalid_raw_relations': [{'relation_index': i, 'relation': rel}
                  for i, rel in enumerate(raw['relations'])
                  if rel['head'] not in {s['token_end'] for s in raw['spans']}
                  or rel['child'] not in {s['token_end'] for s in raw['spans']}]}
    # These checks fail on overlap, truncation, discontinuity or any lost BIO boundary.
    for mode in ('act', 'arg'):
        make_span_examples([record], mode)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, default=Path('data/procedural/ricoreca/raw'))
    p.add_argument('--output', type=Path, default=Path('data/procedural/ricoreca/converted'))
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite converted dataset')
    released, sources = published_records(args.raw_root)
    raw = {}
    for meal in ('dinner', 'lunch'):
        paths = list(args.raw_root.rglob('Annotator_P/prodigy_' + meal + '.json'))
        if len(paths) != 1:
            raise ValueError('Missing or ambiguous primary annotation file')
        path = paths[0]
        rows = json.loads(path.read_text(encoding='utf-8-sig'))
        for row in rows:
            doc_id = row['meta']['id']
            if doc_id in raw:
                raise ValueError('Duplicate primary document ID')
            raw[doc_id] = row
        sources.append({'path': str(path.relative_to(args.raw_root)), 'sha256': sha(path), 'records': len(rows)})
    records = [convert_record(raw[k], released[k]) for k in sorted(released)]
    manifest = make_split(records, args.seed, expected_documents=156)
    manifest.update({'corpus': 'Union of author-released Cross_Val CSV IDs; their folds are not reused',
                     'source_files': sources, 'excluded_raw_ids_outside_published_corpus': sorted(set(raw)-set(released)),
                     'raw_primary_documents': len(raw), 'annotation_conversion_validated': True,
                     'split_script_sha256': sha(Path(__file__).with_name('prepare_ricoreca_split.py'))})
    args.output.mkdir(parents=True)
    counts, hashes = {}, {}
    by_id = {r['id']: r for r in records}
    for split, ids in manifest['document_ids'].items():
        group = [dict(by_id[k], split=split) for k in ids]
        path = args.output / (split + '.jsonl')
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in group), encoding='utf-8')
        hashes[split] = sha(path)
        counts[split] = {'documents': len(group), 'actions': sum(len(r['actions']) for r in group),
                         'linked_arguments': sum(len(a['explicit_arguments']) for r in group for a in r['actions'])}
    audit = {'status': 'passed', 'counts': counts, 'dataset_sha256': hashes,
             'raw_primary_documents': len(raw), 'published_documents': len(records),
             'max_document_tokens': max(len(r['words']) for r in records),
             'action_capacity': 600, 'argument_capacity': 100,
             'unresolved_memberships': 0, 'ambiguous_action_occurrences': 0, 'truncated_spans': 0,
             'all_BIO_roundtrips_exact': True, 'source_text_reconstruction_exact': True,
             'excluded_control_member_mentions': sum(len(r['excluded_control_members']) for r in records),
             'invalid_raw_relations_ignored': sum(len(r['invalid_raw_relations']) for r in records),
             'argument_types': dict(collections.Counter(a['source_type'] for r in records for x in r['actions'] for a in x['explicit_arguments'])),
             'gold_scope': 'Published ACTION spans and their explicit textual member spans, including states/measures/settings/why; source types are not predicted. IF/UNTIL/REPEAT and their exclusive members are outside this action extraction task. No implicit/antecedent imputation.',
             'association': 'Published membership names plus original Prodigy Member/Or/Modifier paths, stopping at predicate boundaries, with Join sharing; Dependency never traversed; ambiguous occurrences cause failure.',
             'source_sha256': {n: sha(Path(__file__).with_name(n)) for n in ('prepare_ricoreca_data.py', 'prepare_ricoreca_split.py', 'prepare_procedural_data.py', 'risec_spans.py')}}
    for filename, obj in [('split.json', manifest), ('conversion_audit.json', audit)]:
        (args.output / filename).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
