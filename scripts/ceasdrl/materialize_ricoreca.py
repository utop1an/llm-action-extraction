"""Extract a local RiCoRecA ZIP with path, content and overwrite checks."""
import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=Path('data/data.zip'))
    p.add_argument('--output', type=Path, default=Path('data/procedural/ricoreca/raw'))
    args = p.parse_args()
    destination = args.output.resolve()
    members, used = [], set()
    with zipfile.ZipFile(args.archive) as archive:
        for info in archive.infolist():
            rel = PurePosixPath(info.filename)
            if rel.is_absolute() or '..' in rel.parts or ':' in str(rel) or '\\' in str(rel):
                raise ValueError('Unsafe ZIP path: ' + info.filename)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('ZIP symlinks are not supported')
            if info.is_dir():
                continue
            key = str(rel).casefold()
            if key in used:
                raise ValueError('Duplicate ZIP file path: ' + info.filename)
            used.add(key)
            target = destination.joinpath(*rel.parts).resolve()
            if destination not in target.parents:
                raise ValueError('ZIP member escapes destination')
            blob = archive.read(info)
            if target.exists() and target.read_bytes() != blob:
                raise ValueError('Refusing to overwrite modified raw data: ' + str(target))
            members.append((info.filename, target, blob))
        for name, target, blob in members:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(blob)
    receipt = {'status': 'materialized', 'archive': str(args.archive),
               'archive_sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest(),
               'archive_bytes': args.archive.stat().st_size,
               'files': [{'path': name, 'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()}
                         for name, _, blob in members],
               'source': 'User supplied official RiCoRecA data archive; online version metadata unavailable.'}
    destination.parent.mkdir(parents=True, exist_ok=True)
    (destination.parent / 'archive_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({'status': receipt['status'], 'files': len(members), 'archive_sha256': receipt['archive_sha256']}))


if __name__ == '__main__':
    main()
