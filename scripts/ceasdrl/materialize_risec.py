"""Materialize a pinned public RISeC bundle after checking every Git blob hash."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path('data/procedural/risec/source_bundle.json'))
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text(encoding='utf-8'))
    destination = args.bundle.parent / 'upstream'
    paths = set()
    for item in bundle['files']:
        relative = PurePosixPath(item['path'])
        if relative.is_absolute() or '..' in relative.parts or ':' in str(relative):
            raise ValueError('Unsafe source path')
        if str(relative) in paths:
            raise ValueError('Duplicate source path')
        paths.add(str(relative))
        raw = item['content'].encode('utf-8')
        digest = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if digest != item['sha'] or len(raw) != item['size']:
            raise ValueError('Source hash mismatch: ' + str(relative))
    for item in bundle['files']:
        path = destination.joinpath(*PurePosixPath(item['path']).parts)
        raw = item['content'].encode('utf-8')
        if path.exists() and path.read_bytes() != raw:
            raise ValueError('Refusing to replace modified source: ' + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    orphans = sorted(p for p in paths if p.endswith('.ann') and p[:-4] + '.txt' not in paths)
    report = {'repository': bundle['repository'], 'revision': bundle['revision'],
              'files_verified': len(paths), 'git_blob_hashes_match': True,
              'orphan_annotations_excluded': orphans,
              'split_documents': {s: sum(p.startswith('data/' + s + '/') and p.endswith('.txt') for p in paths)
                                  for s in ['train', 'dev', 'test']}}
    (args.bundle.parent / 'source_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
