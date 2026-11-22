"""Freeze a document-level RiCoRecA 64/16/20 split from real primary annotations.

This creates a split manifest, not converted BIO targets or trained models.
Like cEASDRL, reserve 20% test, then reserve 20% of the remainder for dev.
"""
import argparse
import hashlib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_primary_records(raw_root):
    raw_root = Path(raw_root)
    if not raw_root.is_dir():
        raise FileNotFoundError('RiCoRecA raw data is missing: ' + str(raw_root))
    records, sources = [], []
    for meal in ('dinner', 'lunch'):
        paths = [p for p in raw_root.rglob('prodigy_' + meal + '.json')
                 if p.parent.name == 'Annotator_P']
        if len(paths) != 1:
            raise ValueError('Expected exactly one Annotator_P/prodigy_' + meal + '.json; found ' + str(len(paths)))
        path = paths[0]
        rows = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(rows, list):
            raise ValueError('Expected a Prodigy JSON list: ' + str(path))
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get('meta'), dict):
                raise ValueError('Missing recipe metadata: ' + str(path))
            doc_id = row['meta'].get('id')
            text = row.get('text')
            if not isinstance(doc_id, str) or not doc_id.strip() or not isinstance(text, str) or not text.strip():
                raise ValueError('Invalid document ID or text in ' + str(path))
            if not isinstance(row.get('spans'), list) or not isinstance(row.get('relations'), list):
                raise ValueError('Missing entity/relation annotations for ' + doc_id)
            records.append({'id': doc_id, 'text': text, 'source': str(path.relative_to(raw_root))})
        sources.append({'path': str(path.relative_to(raw_root)), 'sha256': sha(path), 'records': len(rows)})
    return records, sources


def make_split(records, seed=1, expected_documents=156):
    by_id, by_text = {}, {}
    for row in records:
        doc_id = row['id']
        if doc_id in by_id:
            raise ValueError('Duplicate document ID; adjudicate before splitting: ' + doc_id)
        text_hash = hashlib.sha256(' '.join(row['text'].split()).casefold().encode('utf-8')).hexdigest()
        if text_hash in by_text:
            raise ValueError('Duplicate normalized text; group or adjudicate before splitting: ' + doc_id + ' / ' + by_text[text_hash])
        by_id[doc_id], by_text[text_hash] = row, doc_id
    if len(by_id) != expected_documents:
        raise ValueError('Expected {} primary documents, found {}; no samples were silently skipped'.format(expected_documents, len(by_id)))
    # Sorting fixes the input order independently of file enumeration or JSON order.
    train_dev, test = train_test_split(sorted(by_id), test_size=.2, random_state=seed)
    train, dev = train_test_split(train_dev, test_size=.2, random_state=seed)
    splits = {s: sorted(ids) for s, ids in [('train', train), ('dev', dev), ('test', test)]}
    all_ids = [doc_id for ids in splits.values() for doc_id in ids]
    assert len(all_ids) == len(set(all_ids)) == len(by_id)
    return {
        'status': 'split_frozen', 'dataset': 'RiCoRecA', 'annotation_set': 'Annotator_P',
        'protocol': 'document_holdout_64_16_20', 'split_seed': seed,
        'algorithm': 'sorted IDs; sklearn train_test_split(test_size=0.2, random_state=seed), twice',
        'nominal_fractions': {'train': .64, 'dev': .16, 'test': .20},
        'counts': {s: len(ids) for s, ids in splits.items()},
        'actual_fractions': {s: len(ids) / len(by_id) for s, ids in splits.items()},
        'document_ids': splits,
        'text_sha256': {k: hashlib.sha256(by_id[k]['text'].encode('utf-8')).hexdigest() for k in sorted(by_id)},
        'duplicate_ids': 0, 'duplicate_normalized_texts': 0,
        'annotation_conversion_validated': False, 'training_started': False,
        'note': 'Split only. Check span/relation projection, spreadsheets and context coverage before training.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=Path('data/procedural/ricoreca/raw'))
    parser.add_argument('--output', type=Path, default=Path('data/procedural/ricoreca/split_64_16_20_seed1.json'))
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--expected-documents', type=int, default=156)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite a frozen split: ' + str(args.output))
    records, sources = load_primary_records(args.raw_root)
    from prepare_ricoreca_data import published_records
    released, published_sources = published_records(args.raw_root)
    excluded = sorted(r['id'] for r in records if r['id'] not in released)
    records = [r for r in records if r['id'] in released]
    if set(released) != {r['id'] for r in records} or any(r['text'] != released[r['id']]['recipe'] for r in records):
        raise ValueError('Raw/published corpus mismatch')
    manifest = make_split(records, args.seed, args.expected_documents)
    manifest['source_files'] = sources + published_sources
    manifest['excluded_raw_ids_outside_published_corpus'] = excluded
    manifest['split_script_sha256'] = sha(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(manifest, out, ensure_ascii=False, indent=2)
    print(json.dumps({'status': manifest['status'], 'counts': manifest['counts'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
