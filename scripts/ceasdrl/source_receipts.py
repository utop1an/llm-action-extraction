"""Check source receipts across the scripts/ceasdrl move and Git checkout EOLs."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def matches_source(recorded_name, expected):
    relative = Path(recorded_name)
    if relative.parent not in (Path("."), Path("scripts"), Path("scripts/ceasdrl")):
        return False
    source = HERE / relative.name
    data = source.read_bytes()
    digest = hashlib.sha256
    if digest(data).hexdigest() == expected:
        return True
    lf = data.replace(b"\r\n", b"\n")
    current = digest(lf).hexdigest()
    if expected in (current, digest(lf.replace(b"\n", b"\r\n")).hexdigest()):
        return True
    versions = json.loads((HERE / "source_relocations.json").read_text(encoding="utf-8"))
    pair = versions.get(relative.name)
    return bool(pair and current == pair["relocated_lf"]
                and expected in (pair["original_lf"], pair["original_crlf"]))
