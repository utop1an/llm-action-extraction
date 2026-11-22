"""Repair explicitly authorized failures in the frozen GPT-5.4 repeats.

Pause the verified original worker before advance/evaluate: these commands take
the SAME worker.lock. No daemon is started. Original manifests/results and frozen
sources are never edited. --submit is required to create an inference batch.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_nl2p_repeats_batch as original

batch = original.batch
DIRECTORY = original.DIRECTORY / "repairs"
MODEL = "gpt-5.4"
SNAPSHOT = original.MODELS[MODEL]
SERVER_CODES = {"server_error", "internal_server_error", "internal_error",
                "service_unavailable", "service_unavailable_error", "server_is_overloaded"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(6):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.2 * (attempt + 1))


def immutable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Immutable repair artifact changed: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(content)


def input_lines(manifest):
    data = Path(manifest["input_jsonl"]).read_bytes()
    if sha(data) != manifest["input_sha256"]:
        raise ValueError("Original input hash changed")
    result = {}
    for line in data.splitlines(keepends=True):
        if not line.strip():
            continue
        row = json.loads(line)
        cid = row["custom_id"]
        if cid in result or row["body"]["model"] != SNAPSHOT:
            raise ValueError("Duplicate request or changed model snapshot")
        result[cid] = line
    if set(result) != {r["custom_id"] for r in manifest["records"]}:
        raise ValueError("Original input/record membership mismatch")
    return result


def successful(row):
    response = row.get("response") or {}
    return response.get("status_code") == 200 and not row.get("error") and not (response.get("body") or {}).get("error")


def server_failure(row):
    """HTTP 200/4xx, absent output, expiry, and parse failures never qualify."""
    response = row.get("response") or {}
    status = response.get("status_code")
    if isinstance(status, int):
        return 500 <= status < 600
    error = row.get("error") or {}
    return error.get("code") in SERVER_CODES or error.get("type") in SERVER_CODES


def credit_balance_failure(row):
    """A credit exhaustion 429 is distinct from rate limits and server errors."""
    response = row.get("response") or {}
    error = (response.get("body") or {}).get("error") or {}
    return (response.get("status_code") == 429
            and error.get("type") == "insufficient_quota"
            and error.get("code") == "credit_balance_exhausted")


def retry_reason(row, policy=None):
    policy = policy if policy is not None else {"server_failure_retries_authorized": True}
    if policy.get("server_failure_retries_authorized") and server_failure(row):
        return "server_failure"
    if (policy.get("credit_balance_retries_authorized")
            and policy.get("credits_replenished_confirmed") and credit_balance_failure(row)):
        return "credit_balance_exhausted_429"
    return None


def load_policy(repeat, condition):
    policy = batch.load_manifest(DIRECTORY / "authorization.json")
    if (policy.get("campaign") != original.CAMPAIGN or policy.get("model") != MODEL
            or policy.get("model_snapshot") != SNAPSHOT
            or repeat not in policy.get("rounds", [])
            or condition not in policy.get("conditions", [])
            or repeat not in (1, 2, 3) or condition not in ("minimal", "full")):
        raise ValueError("Repair authorization scope mismatch")
    if policy.get("script_sha256") != sha(Path(__file__).read_bytes()):
        raise ValueError("Repair script hash differs from validated authorization")
    if not (policy.get("server_failure_retries_authorized")
            or policy.get("credit_balance_retries_authorized")):
        raise ValueError("Repair authorization is missing")
    return policy


def check_submission_hold():
    path = original.DIRECTORY / "submission_hold.json"
    if path.exists() and batch.load_manifest(path).get("active"):
        raise ValueError("Submission hold is active; no upload or batch creation allowed")


def index_rows(rows, expected_ids):
    indexed = {}
    for row in rows:
        cid = row.get("custom_id")
        if cid not in expected_ids or cid in indexed:
            raise ValueError("Unexpected or duplicate response custom_id")
        if successful(row) and (row["response"].get("body") or {}).get("model") != SNAPSHOT:
            raise ValueError("Response model snapshot mismatch")
        indexed[cid] = row
    if set(indexed) != set(expected_ids):
        raise ValueError("Missing request-level evidence; do not infer failure")
    return indexed


def apply_attempt(latest, selected, origins, rows, bid, expected_ids, is_original=False, policy=None):
    if not is_original:
        if set(expected_ids) & set(selected):
            raise ValueError("A successful sample was resubmitted")
        if any(cid not in latest or not retry_reason(latest[cid], policy) for cid in expected_ids):
            raise ValueError("Retry lacks confirmed authorized failure evidence")
    for cid, row in index_rows(rows, set(expected_ids)).items():
        latest[cid] = row
        if successful(row):
            if cid in selected:
                raise ValueError("Duplicate success; never choose best-of outputs")
            selected[cid] = row
            origins[cid] = {"batch_id": bid, "row_sha256": sha(json.dumps(row, sort_keys=True).encode())}


def fetch(client, manifest, expected_ids, cache):
    remote = client.batches.retrieve(manifest["batch_id"]).model_dump()
    if remote["input_file_id"] != manifest["input_file_id"]:
        raise ValueError("Remote input file differs from saved submission")
    folder = cache / remote["id"]
    save(folder / "remote.json", remote)
    # A model-wide capacity rejection never executed any of this attempt.
    if original.capacity_rejected(remote):
        counts = remote.get("request_counts") or {}
        if any(counts.get(k) != 0 for k in ["total", "completed", "failed"]):
            raise ValueError("Capacity rejection counts are not explicitly zero")
        return remote, None
    if remote["status"] not in original.TERMINAL:
        return remote, None
    rows = []
    hashes = {}
    recorded_hashes = batch.load_manifest(folder / 'file_hashes.json') if (folder / 'file_hashes.json').exists() else {}
    for label in ["output", "error"]:
        fid = remote.get(label + "_file_id")
        if fid:
            target = folder / (label + ".jsonl")
            # Re-fetch if needed; immutable comparison protects cached evidence.
            content = client.files.content(fid).read() if not target.exists() else target.read_bytes()
            if label in recorded_hashes and recorded_hashes[label] != {"file_id": fid, "sha256": sha(content)}:
                raise ValueError("Cached remote evidence changed")
            immutable(target, content)
            hashes[label] = {"file_id": fid, "sha256": sha(content)}
            rows.extend(json.loads(line) for line in content.splitlines() if line.strip())
    index_rows(rows, set(expected_ids))
    if (remote.get("request_counts") or {}).get("total") != len(expected_ids):
        raise ValueError("Remote request denominator mismatch")
    save(folder / "file_hashes.json", hashes)
    return remote, rows


def prepare(base, lines, latest, selected, origins, folder, number, source_batch_ids, policy=None):
    eligible = [cid for cid in lines if cid not in selected and retry_reason(latest[cid], policy)]
    if not eligible:
        raise ValueError("No confirmed authorized failures to retry")
    target = folder / "attempts" / f"{number:03d}"
    input_path = target / "input.jsonl"
    payload = b"".join(lines[cid] for cid in eligible)
    immutable(input_path, payload)
    path = target / "manifest.json"
    if path.exists():
        existing = batch.load_manifest(path)
        if existing["input_sha256"] != sha(payload):
            raise ValueError("Prepared attempt membership changed")
        return path
    keep = ["solver", "prompt_name", "model", "model_name", "datasets", "limit",
            "temperature", "request_parameters", "dataset_sha256", "coref", "coref_dir", "endpoint"]
    manifest = {k: deepcopy(base[k]) for k in keep if k in base}
    manifest.update(run_id=f"{base['run_id']}-authorized-repair-{number:03d}",
                    input_jsonl=str(input_path.resolve()), input_sha256=sha(payload),
                    records=[r for r in base["records"] if r["custom_id"] in eligible],
                    request_count=len(eligible), created_at=original.now(),
                    original_input_sha256=base["input_sha256"], original_batch_id=base["batch_id"],
                    source_batch_ids=source_batch_ids,
                    success_ids_excluded=sorted(selected),
                    failure_evidence={cid: latest[cid] for cid in eligible},
                    retry_reasons={cid: retry_reason(latest[cid], policy) for cid in eligible},
                    authorization_sha256=sha((DIRECTORY / 'authorization.json').read_bytes()),
                    authorization_snapshot=deepcopy(policy),
                    request_line_sha256={cid: sha(lines[cid]) for cid in eligible})
    save(path, manifest)
    return path


def verify_prepared(manifest, lines, latest, selected, policy=None):
    payload = Path(manifest["input_jsonl"]).read_bytes()
    ids = [r["custom_id"] for r in manifest["records"]]
    if len(ids) != len(set(ids)) or sha(payload) != manifest["input_sha256"]:
        raise ValueError("Prepared retry hash/IDs changed")
    if payload != b"".join(lines[cid] for cid in lines if cid in set(ids)):
        raise ValueError("Retry request bytes differ from original")
    if any(cid in selected or cid not in latest or not retry_reason(latest[cid], policy) for cid in ids):
        raise ValueError("Prepared retry contains a success or unconfirmed failure")
    for cid in ids:
        if manifest.get('failure_evidence', {}).get(cid) != latest[cid]:
            raise ValueError("Prepared retry failure evidence changed")
        if manifest.get('retry_reasons', {}).get(cid) != retry_reason(latest[cid], policy):
            raise ValueError("Prepared retry authorization reason changed")
    return ids


def capacity_released(client, rejected, release_id):
    if release_id:
        released = client.batches.retrieve(release_id).model_dump()
        return (released["status"] == "completed"
                and (released.get("metadata") or {}).get("model") in {MODEL, SNAPSHOT}
                and released["completed_at"] > rejected["created_at"])
    return not any(x.status in original.ACTIVE and (x.metadata or {}).get("model") in {MODEL, SNAPSHOT}
                   for x in client.batches.list(limit=100))


def assemble(base, selected, origins, folder):
    if set(selected) != {r["custom_id"] for r in base["records"]}:
        raise ValueError("Cannot assemble an incomplete run")
    target = folder / "merged"
    merged = b"".join((json.dumps(selected[r["custom_id"]], ensure_ascii=False) + "\n").encode()
                      for r in base["records"])
    immutable(target / "output.jsonl", merged)
    path = target / "manifest.json"
    if not path.exists():
        manifest = deepcopy(base)
        for key in ["collected_at", "output_jsonl", "result_dir", "failure_count"]:
            manifest.pop(key, None)
        manifest.update(batch_status="completed", result_solver=base["result_solver"] + "_server_repaired",
                        assembled_from_batches=sorted({v['batch_id'] for v in origins.values()}),
                        original_manifest=str(Path(base["input_jsonl"]).parent / "manifest.json"),
                        assembly_kind="original successes plus first successful authorized failure repairs")
        save(path, manifest)
    manifest = batch.load_manifest(path)
    if not manifest.get("collected_at"):
        batch.collect(SimpleNamespace(manifest=str(path), output_jsonl=str(target / "output.jsonl"),
                                      output_file_id=None, result_solver=None))
    manifest = batch.load_manifest(path)
    save(target / "provenance.json", {"output_sha256": sha(merged), "samples": origins,
                                       "request_count": len(selected), "parse_failure_count": manifest["failure_count"]})
    return str(path.resolve()), manifest["failure_count"]


def advance(repeat, condition, submit=False, release_id=None):
    matrix = batch.load_manifest(original.MATRIX)
    original.verify_frozen(matrix)
    policy = load_policy(repeat, condition)
    if submit:
        check_submission_hold()
    run = next(r for r in matrix["runs"] if r['model'] == MODEL and r['repeat'] == repeat and r['condition'] == condition)
    base = batch.load_manifest(Path(run["manifest"]))
    if base["model_name"] != SNAPSHOT or base["request_count"] != 420:
        raise ValueError("Unexpected original experiment")
    lines = input_lines(base)
    if len(lines) != 420 or dict(Counter(r['dataset'] for r in base['records'])) != original.COUNTS:
        raise ValueError("Original dataset coverage is not 420 documents")
    folder = DIRECTORY / f"r{repeat}-{condition}"
    client = batch.get_openai_client(MODEL).with_options(timeout=45, max_retries=0)
    if str(client.base_url).rstrip('/') != 'https://api.openai.com/v1':
        raise ValueError("Unexpected API endpoint")
    latest, selected, origins = {}, {}, {}
    source_ids = []
    report = {"repeat": repeat, "condition": condition, "checked_at": original.now(),
              "original_batch_id": base["batch_id"], "expected_requests": 420}

    def finish(status, **extra):
        report.update(status=status, successes=len(selected), **extra)
        save(folder / "state.json", report)
        print(json.dumps(report, ensure_ascii=False))
        return report

    def renewed_credit_exhaustion(remote, rows):
        # The user confirmed a replenishment for the recorded failed batches.
        # A newly exhausted balance must not trigger a blind resubmission loop.
        if (policy.get('credit_balance_retries_authorized')
                and remote['id'] not in policy.get('credit_balance_recovery_source_batch_ids', [])
                and any(credit_balance_failure(row) for row in rows)):
            hold_path = original.DIRECTORY / 'submission_hold.json'
            previous = batch.load_manifest(hold_path) if hold_path.exists() else None
            if not previous or not previous.get('active'):
                save(hold_path, {'active': True, 'at': original.now(),
                    'reason': 'New credit_balance_exhausted 429 after the recorded replenishment',
                    'batch_id': remote['id'], 'previous_hold': previous,
                    'resume_requires': ['New direct user confirmation of replenished API credits']})
            return True
        return False

    remote, rows = fetch(client, base, lines, folder / "raw")
    if rows is None:
        return finish("await_original_terminal", remote_status=remote['status'])
    apply_attempt(latest, selected, origins, rows, remote['id'], lines, is_original=True)
    source_ids.append(remote['id'])
    if renewed_credit_exhaustion(remote, rows):
        return finish('requires_attention_renewed_credit_exhaustion', batch_id=remote['id'])
    attempts = sorted((folder / "attempts").glob('*/manifest.json'))
    pending_path = None
    capacity_retry_of = None
    for i, path in enumerate(attempts):
        manifest = batch.load_manifest(path)
        ids = verify_prepared(manifest, lines, latest, selected, policy)
        if not manifest.get('batch_id'):
            if i != len(attempts)-1:
                raise ValueError("Unresolved submission precedes another attempt")
            pending_path = path
            break
        remote, rows = fetch(client, manifest, ids, folder / 'raw')
        if rows is None:
            if original.capacity_rejected(remote):
                if i != len(attempts)-1:
                    continue  # retained zero-execution attempt
                if not submit or not capacity_released(client, remote, release_id):
                    return finish('await_repair_capacity', batch_id=remote['id'])
                capacity_retry_of = remote['id']
                break
            return finish('repair_in_progress', batch_id=remote['id'], request_counts=remote.get('request_counts'))
        apply_attempt(latest, selected, origins, rows, remote['id'], ids, policy=policy)
        source_ids.append(remote['id'])
        if renewed_credit_exhaustion(remote, rows):
            return finish('requires_attention_renewed_credit_exhaustion', batch_id=remote['id'])
    eligible = [cid for cid in lines if cid not in selected and retry_reason(latest[cid], policy)]
    blocked = [cid for cid in lines if cid not in selected and not retry_reason(latest[cid], policy)]
    report.update(eligible_server_failures=sum(server_failure(latest[cid]) for cid in eligible),
                  eligible_credit_balance_failures=sum(credit_balance_failure(latest[cid]) for cid in eligible),
                  eligible_authorized_failures=len(eligible), blocked_nonserver_ids=blocked,
                  blocked_unauthorized_ids=blocked)
    if len(selected) == 420:
        merged, parse_failures = assemble(base, selected, origins, folder)
        return finish('complete' if not parse_failures else 'requires_attention_parse_failure',
                      merged_manifest=merged, parse_failure_count=parse_failures)
    if not eligible:
        return finish('requires_attention_nonserver_failure')
    pending_path = pending_path or prepare(base, lines, latest, selected, origins, folder,
                                           len(attempts)+1, source_ids, policy)
    manifest = batch.load_manifest(pending_path)
    verify_prepared(manifest, lines, latest, selected, policy)
    if capacity_retry_of:
        manifest.update(capacity_retry_of=capacity_retry_of, capacity_release_trigger=release_id or 'no_active_model_batches')
        save(pending_path, manifest)
    if not submit:
        return finish('prepared', repair_manifest=str(pending_path.resolve()), retry_count=manifest['request_count'])
    # Existing submit persists file ID and submission intent before batch create,
    # and recovers ambiguous submissions by input_file_id instead of duplicating.
    check_submission_hold()
    batch.submit(SimpleNamespace(manifest=str(pending_path)))
    manifest = batch.load_manifest(pending_path)
    return finish('repair_submitted', repair_manifest=str(pending_path.resolve()),
                  batch_id=manifest['batch_id'], retry_count=manifest['request_count'])


def evaluate(adjusted):
    matrix = batch.load_manifest(original.MATRIX)
    original.verify_frozen(matrix)
    runs = []
    for run in matrix['runs']:
        if run['model'] != MODEL:
            continue
        state = batch.load_manifest(DIRECTORY / f"r{run['repeat']}-{run['condition']}" / 'state.json')
        if state['status'] != 'complete' or state['successes'] != 420:
            raise ValueError('All six runs must be complete before comparing')
        manifest = batch.load_manifest(Path(state['merged_manifest']))
        if manifest['failure_count'] != 0:
            raise ValueError('Unresolved parsing failures')
        runs.append({**run, 'manifest': state['merged_manifest']})
    repaired = {**matrix, 'runs': runs, 'repair_authorization': str(DIRECTORY / 'authorization.json')}
    save(DIRECTORY / 'matrix.json', repaired)
    for relative in ['evaluation.py', 'src/evaluation_helpers.py']:
        immutable(DIRECTORY / 'source_snapshot' / relative,
                  (original.DIRECTORY / 'source_snapshot' / relative).read_bytes())
    from scripts import evaluate_nl2p_repeats as evaluator
    evaluator.MATRIX = DIRECTORY / 'matrix.json'
    evaluator.CAMPAIGN = original.CAMPAIGN + '/repaired'
    evaluator.main(MODEL, adjusted=adjusted)
    target = ROOT / 'results' / evaluator.CAMPAIGN / MODEL / ('comparison_adjusted.json' if adjusted else 'comparison.json')
    report = batch.load_manifest(target)
    report['authorized_failure_recovery'] = {
        'authorization': str(DIRECTORY / 'authorization.json'),
        'selection': 'Keep original successes; retry only explicitly authorized confirmed failures; distinguish server errors from credit exhaustion 429; retain first success, never best-of.',
        'runs': [{'repeat': r['repeat'], 'condition': r['condition'], 'merged_manifest': r['manifest'],
                  'sample_provenance': str(Path(r['manifest']).parent / 'provenance.json')} for r in runs],
        'limitation': 'Repairs occur at later times; disclose repair counts and timing with the three-repeat comparison.'}
    save(target, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['advance', 'evaluate'])
    parser.add_argument('--repeat', type=int, choices=[1, 2, 3])
    parser.add_argument('--condition', choices=['minimal', 'full'])
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--released-batch-id')
    parser.add_argument('--adjusted', action='store_true')
    args = parser.parse_args()
    if args.command == 'advance' and (args.repeat is None or args.condition is None):
        parser.error('advance requires --repeat and --condition')
    import msvcrt
    with (original.DIRECTORY / 'worker.lock').open('r+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        if args.command == 'advance':
            advance(args.repeat, args.condition, args.submit, args.released_batch_id)
        else:
            evaluate(args.adjusted)
