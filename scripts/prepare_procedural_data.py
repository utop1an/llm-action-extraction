"""Convert public procedural annotations without inventing action-argument links.

Python 3.6 compatible. RISeC retains its official document split. ReciFineGold
has action spans but no argument links and must not be used for argument F1.
"""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path


TOKEN_RE = re.compile(r"\w+(?:['’-]\w+)*|[^\w\s]", re.UNICODE)


def tokenize(text, split_hyphens=False):
    pattern = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE) if split_hyphens else TOKEN_RE
    return [(m.group(), m.start(), m.end()) for m in pattern.finditer(text)]


def span_tokens(tokens, spans):
    result = []
    for start, end in spans:
        matches = [i for i, (_, a, b) in enumerate(tokens) if a < end and b > start]
        if not matches or tokens[matches[0]][1] != start or tokens[matches[-1]][2] != end:
            raise ValueError("Annotation boundary is not aligned with tokens: {}".format((start, end)))
        result.extend(matches)
    return sorted(set(result))


def read_risec(txt_path, split):
    # newline='' preserves the character offsets used by BRAT.
    with txt_path.open(encoding='utf-8', newline='') as f:
        text = f.read()
    tokens = tokenize(text)
    entities, relations, notes = {}, [], {}
    for line in txt_path.with_suffix('.ann').read_text(encoding='utf-8').splitlines():
        fields = line.split('\t')
        if line.startswith('T'):
            tag, positions = fields[1].split(' ', 1)
            spans = [tuple(map(int, part.split())) for part in positions.split(';')]
            actual = ' '.join(text[a:b] for a, b in spans)
            if actual != fields[2]:
                raise ValueError('BRAT text mismatch {} {}'.format(txt_path, fields[0]))
            entities[fields[0]] = {'id': fields[0], 'type': tag, 'spans': spans,
                                   'text': actual, 'token_indices': span_tokens(tokens, spans)}
        elif line.startswith('R'):
            role, source, target = fields[1].split()
            relations.append({'id': fields[0], 'role': role,
                              'source': source.split(':')[1], 'target': target.split(':')[1]})
        elif line.startswith('#'):
            notes[fields[1].split()[-1]] = fields[2]
    actions = []
    for entity in entities.values():
        if entity['type'] != 'AC':
            continue
        explicit, implicit, excluded = [], [], []
        for rel in relations:
            if rel['source'] != entity['id']:
                continue
            target = entities[rel['target']]
            item = dict(rel, target_text=target['text'], token_indices=target['token_indices'])
            if rel['role'].endswith('_X') or target['type'] == 'AC':
                item['annotation_note'] = notes.get(rel['id'])
                implicit.append(item)
            elif target['type'] == 'CONDITION_CLAUSE':
                excluded.append(item)
            else:
                explicit.append(item)
        actions.append({'source_id': entity['id'], 'trigger_tokens': entity['token_indices'],
                        'text': entity['text'], 'explicit_arguments': explicit,
                        'implicit_arguments': implicit, 'excluded_conditions': excluded})
    actions.sort(key=lambda a: a['trigger_tokens'][0])
    return {'id': txt_path.stem, 'source': 'RISeC', 'split': split, 'text': text,
            'words': [t[0] for t in tokens], 'token_offsets': [list(t[1:]) for t in tokens],
            'actions': actions, 'argument_links_available': True,
            'annotations': list(entities.values()), 'relations': relations}


def read_recifinegold(path, seed=0):
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    # Group exact normalized text duplicates before assigning a document split.
    groups = collections.defaultdict(list)
    for row in records:
        norm = ' '.join(row['text'].split()).casefold()
        groups[norm].append(row)
    keys = sorted(groups, key=lambda key: hashlib.sha256((str(seed) + key).encode()).hexdigest())
    splits = {key: ('train' if i < int(.8 * len(keys)) else
                    'dev' if i < int(.9 * len(keys)) else 'test') for i, key in enumerate(keys)}
    converted = []
    for key in keys:
        for row in groups[key]:
            tokens = tokenize(row['text'])
            actions = []
            for label in row['label']:
                start, end = int(label['start']), int(label['end'])
                if not 0 <= start < end <= len(row['text']):
                    raise ValueError('Invalid offsets in {}'.format(row['id']))
                if label['tag'] == 'Action by chef':
                    actions.append({'trigger_tokens': span_tokens(tokens, [(start, end)]),
                                    'text': row['text'][start:end],
                                    'explicit_arguments': None, 'implicit_arguments': None})
            actions.sort(key=lambda a: a['trigger_tokens'][0])
            converted.append({'id': row['id'], 'source': 'ReciFineGold', 'split': splits[key],
                              'text': row['text'], 'words': [t[0] for t in tokens],
                              'token_offsets': [list(t[1:]) for t in tokens], 'actions': actions,
                              'argument_links_available': False, 'annotations': row['label']})
    return converted


def read_openpi_a(path, seed=0):
    """Project released silver spans; do not call these exhaustive action labels.

    The release has one selected trigger per step, not all executable actions.
    Group by original document AND repeated sentence before splitting.
    """
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    parent = {str(r['doc_id']): str(r['doc_id']) for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    text_doc = {}
    for row in rows:
        text = ' '.join(row['sentence'].split()).casefold()
        doc = str(row['doc_id'])
        if text in text_doc:
            a, b = find(doc), find(text_doc[text])
            parent[max(a, b)] = min(a, b)
        text_doc[text] = doc
    groups = sorted(set(find(d) for d in parent),
                    key=lambda d: hashlib.sha256((str(seed) + ':' + d).encode()).hexdigest())
    splits = {d: ('train' if i < int(.8 * len(groups)) else
                  'dev' if i < int(.9 * len(groups)) else 'test') for i, d in enumerate(groups)}
    result = []
    for row in rows:
        text = row['sentence']
        span = row['span']
        if text[span['start']:span['end']] != span['text']:
            raise ValueError('OpenPI-A span text mismatch')
        # OpenPI-A annotates 'fold' inside 'Re-fold'; use lexical punctuation
        # tokenization rather than silently expanding or dropping the span.
        tokens = tokenize(text, split_hyphens=True)
        action = {'trigger_tokens': span_tokens(tokens, [(span['start'], span['end'])]),
                  'text': span['text'], 'explicit_arguments': None, 'implicit_arguments': None}
        result.append({'id': 'doc{}_step{}'.format(row['doc_id'], row['step_id']),
                       'source_group': str(row['doc_id']), 'source': 'OpenPI-A-silver',
                       'split': splits[find(str(row['doc_id']))], 'text': text,
                       'words': [t[0] for t in tokens], 'token_offsets': [list(t[1:]) for t in tokens],
                       'actions': [action], 'argument_links_available': False,
                       'annotation_scope': 'One selected state-change trigger per step; not exhaustive planning actions.',
                       'annotations': [row], 'topic': row['topic']})
    return result


def audit(records):
    ids = [r['id'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate document IDs')
    text_splits = collections.defaultdict(set)
    source_splits = collections.defaultdict(set)
    for r in records:
        text_splits[' '.join(r['text'].split()).casefold()].add(r['split'])
        source_splits[r.get('source_group', r['id'])].add(r['split'])
    if any(len(s) > 1 for s in text_splits.values()):
        raise ValueError('Identical document text leaks across splits')
    if any(len(s) > 1 for s in source_splits.values()):
        raise ValueError('Original source document leaks across splits')
    return {'documents': len(records), 'split_documents': dict(collections.Counter(r['split'] for r in records)),
            'actions': sum(len(r['actions']) for r in records),
            'multi_token_actions': sum(len(a['trigger_tokens']) > 1 for r in records for a in r['actions']),
            'explicit_argument_links': sum(len(a['explicit_arguments'] or []) for r in records for a in r['actions']),
            'implicit_argument_links': sum(len(a['implicit_arguments'] or []) for r in records for a in r['actions']),
            'max_document_tokens': max((len(r['words']) for r in records), default=0),
            'es_op_ex_available': False,
            'source_documents': len(source_splits),
            'annotation_scopes': sorted(set(r.get('annotation_scope', 'Annotated action spans') for r in records)),
            'projection': 'First annotated token of each action/explicit argument span. All annotations retained.',
            'warning': 'Observed action occurrences are extraction targets, not planning-essential labels.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format', choices=['risec', 'recifinegold', 'openpi-a'], required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.format == 'risec':
        records = [read_risec(p, split) for split in ['train', 'dev', 'test']
                   for p in sorted((args.input / split).glob('*.txt'))]
    elif args.format == 'recifinegold':
        records = read_recifinegold(args.input, args.seed)
    else:
        records = read_openpi_a(args.input, args.seed)
    if not records:
        raise ValueError('No records found')
    stats = audit(records)
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'dev', 'test']:
        with (args.output / (split + '.jsonl')).open('w', encoding='utf-8') as f:
            for row in records:
                if row['split'] == split:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
    (args.output / 'audit.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
