import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import repair_nl2p_repeats_batch as repair


def response(cid, status=200, code=None):
    return {'custom_id': cid, 'response': {'status_code': status, 'body':
        {'model': repair.SNAPSHOT, 'choices': [{'message': {'content': '[]'}}]} if status == 200 else
        {'error': {'type': code or 'server_error'}}}, 'error': None}


@pytest.mark.parametrize('status,expected', [(200, False), (400, False), (429, False),
                                           (500, True), (502, True), (503, True), (504, True)])
def test_only_server_http_failures_qualify(status, expected):
    assert repair.server_failure(response('x', status)) is expected


@pytest.mark.parametrize('code,expected', [('server_error', True), ('server_is_overloaded', True),
    ('batch_expired', False), ('rate_limit_exceeded', False), ('request_timeout', False), (None, False)])
def test_ambiguous_and_nonserver_failures_are_not_retried(code, expected):
    assert repair.server_failure({'response': None, 'error': {'code': code}}) is expected


def test_response_evidence_requires_complete_unique_membership_and_snapshot():
    with pytest.raises(ValueError, match='Missing'):
        repair.index_rows([response('a')], {'a', 'b'})
    with pytest.raises(ValueError, match='duplicate'):
        repair.index_rows([response('a'), response('a')], {'a'})
    with pytest.raises(ValueError, match='Unexpected'):
        repair.index_rows([response('other')], {'a'})
    bad = response('a')
    bad['response']['body']['model'] = 'different-snapshot'
    with pytest.raises(ValueError, match='snapshot'):
        repair.index_rows([bad], {'a'})


def test_success_cannot_be_retried_even_with_an_old_error_record():
    latest = {'a': response('a', 500)}
    selected = {'a': response('a')}
    with pytest.raises(ValueError, match='successful sample'):
        repair.apply_attempt(latest, selected, {}, [response('a')], 'retry', ['a'])


class Remote:
    def __init__(self, data):
        self.data = data
    def model_dump(self):
        return copy.deepcopy(self.data)
    def __getattr__(self, key):
        return self.data.get(key)


class Client:
    base_url = 'https://api.openai.com/v1/'
    def __init__(self):
        self.jobs, self.contents = {}, {}
        self.created = []
        self.lose_create_response = False
        self.batches = SimpleNamespace(retrieve=lambda bid: Remote(self.jobs[bid]),
            list=lambda **kw: [Remote(x) for x in self.jobs.values()], create=self.create_batch)
        self.files = SimpleNamespace(content=lambda fid: SimpleNamespace(read=lambda: self.contents[fid]), create=self.upload)
    def with_options(self, **kw):
        return self
    def upload(self, file, purpose):
        fid = f'input-{len(self.contents)}'
        self.contents[fid] = file.read()
        return SimpleNamespace(id=fid)
    def create_batch(self, input_file_id, endpoint, completion_window, metadata):
        bid = f'repair-{len(self.created)+1}'
        ids = [json.loads(x)['custom_id'] for x in self.contents[input_file_id].splitlines()]
        self.jobs[bid] = {'id': bid, 'status': 'in_progress', 'input_file_id': input_file_id,
            'request_counts': {'completed': 0, 'failed': 0, 'total': len(ids)}, 'errors': None,
            'output_file_id': None, 'error_file_id': None, 'metadata': metadata, 'created_at': 100}
        self.created.append(ids)
        if self.lose_create_response:
            self.lose_create_response = False
            raise TimeoutError('Lost successful create response')
        return Remote(self.jobs[bid])
    def complete(self, bid, rows):
        fid = bid + '-output'
        self.contents[fid] = b''.join((json.dumps(x)+'\n').encode() for x in rows)
        self.jobs[bid].update(status='completed', output_file_id=fid,
            request_counts={'completed': sum(repair.successful(x) for x in rows),
                            'failed': sum(not repair.successful(x) for x in rows), 'total': len(rows)})


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    client = Client()
    records, requests = [], []
    for dataset, count in repair.original.COUNTS.items():
        for i in range(count):
            cid = f'{dataset}:{i}'
            records.append({'custom_id': cid, 'dataset': dataset, 'doc_id': i})
            requests.append({'custom_id': cid, 'method': 'POST', 'url': '/v1/chat/completions',
                'body': {'model': repair.SNAPSHOT, 'messages': [{'role': 'user', 'content': f'Original prompt {cid}'}]}})
    data = b''.join((json.dumps(x, ensure_ascii=False)+'\r\n').encode() for x in requests)
    inp = tmp_path/'original.jsonl'
    inp.write_bytes(data)
    manifest = {'model': repair.MODEL, 'model_name': repair.SNAPSHOT, 'solver': 'nl2p_minimal',
        'result_solver': 'test_original', 'datasets': list(repair.original.COUNTS), 'limit': None,
        'run_id': 'repeat1-minimal', 'endpoint': '/v1/chat/completions', 'records': records,
        'request_count': 420, 'input_jsonl': str(inp), 'input_sha256': repair.sha(data),
        'input_file_id': 'original-input', 'batch_id': 'original'}
    manifest_path = tmp_path/'manifest.json'
    repair.save(manifest_path, manifest)
    matrix_path = tmp_path/'matrix.json'
    repair.save(matrix_path, {'runs': [{'model': repair.MODEL, 'repeat': 1, 'condition': 'minimal', 'manifest': str(manifest_path)}]})
    directory = tmp_path/'repairs'
    repair.save(directory/'authorization.json', {'server_failure_retries_authorized': True,
        'campaign': repair.original.CAMPAIGN, 'model': repair.MODEL, 'model_snapshot': repair.SNAPSHOT,
        'rounds': [1, 2, 3], 'conditions': ['minimal', 'full'],
        'script_sha256': repair.sha(Path(repair.__file__).read_bytes())})
    monkeypatch.setattr(repair, 'DIRECTORY', directory)
    monkeypatch.setattr(repair.original, 'DIRECTORY', tmp_path)
    monkeypatch.setattr(repair.original, 'MATRIX', matrix_path)
    monkeypatch.setattr(repair.original, 'verify_frozen', lambda matrix: None)
    monkeypatch.setattr(repair.batch, 'get_openai_client', lambda model: client)
    def collect(args):
        m = repair.batch.load_manifest(Path(args.manifest))
        m.update(collected_at='now', failure_count=0)
        repair.save(args.manifest, m)
    monkeypatch.setattr(repair.batch, 'collect', collect)
    ids = [r['custom_id'] for r in records]
    client.jobs['original'] = {'id': 'original', 'input_file_id': 'original-input', 'status': 'in_progress',
        'request_counts': {'total': 420, 'completed': 0, 'failed': 0}, 'errors': None,
        'output_file_id': None, 'error_file_id': None, 'metadata': {'model': repair.MODEL}, 'created_at': 1}
    return client, ids, directory, manifest_path


def test_waits_for_original_completion_without_any_submission(campaign):
    client, ids, directory, _ = campaign
    assert repair.advance(1, 'minimal', submit=True)['status'] == 'await_original_terminal'
    assert client.created == []


def test_repeated_server_repair_keeps_successes_and_exact_request_bytes(campaign):
    client, ids, directory, manifest_path = campaign
    original_bytes = manifest_path.read_bytes()
    rows = [response(cid, 500 if i < 2 else 200) for i, cid in enumerate(ids)]
    client.complete('original', rows)
    first = repair.advance(1, 'minimal', submit=True)
    assert first['retry_count'] == 2 and client.created == [ids[:2]]
    repair_manifest = repair.batch.load_manifest(Path(first['repair_manifest']))
    lines = repair.input_lines(repair.batch.load_manifest(manifest_path))
    assert Path(repair_manifest['input_jsonl']).read_bytes() == lines[ids[0]] + lines[ids[1]]
    assert repair.advance(1, 'minimal', submit=True)['status'] == 'repair_in_progress'
    assert len(client.created) == 1
    client.complete('repair-1', [response(ids[0]), response(ids[1], 503)])
    second = repair.advance(1, 'minimal', submit=True)
    assert second['retry_count'] == 1 and client.created[-1] == [ids[1]]
    client.complete('repair-2', [response(ids[1])])
    final = repair.advance(1, 'minimal', submit=True)
    assert final['status'] == 'complete' and final['successes'] == 420
    merged = [json.loads(x) for x in (directory/'r1-minimal/merged/output.jsonl').read_bytes().splitlines()]
    assert merged[2:] == rows[2:]
    assert manifest_path.read_bytes() == original_bytes
    assert len(client.created) == 2
    assert repair.advance(1, 'minimal', submit=True)['status'] == 'complete'
    assert len(client.created) == 2


def test_lost_create_response_recovers_same_attempt_without_duplicate(campaign):
    client, ids, directory, _ = campaign
    client.complete('original', [response(cid, 500 if i == 0 else 200) for i, cid in enumerate(ids)])
    client.lose_create_response = True
    with pytest.raises(TimeoutError):
        repair.advance(1, 'minimal', submit=True)
    assert len(client.created) == 1
    repair.advance(1, 'minimal', submit=True)
    assert len(client.created) == 1
    assert len(list((directory/'r1-minimal/attempts').glob('*/manifest.json'))) == 1


def test_nonserver_failure_and_malformed_success_are_never_retried(campaign):
    client, ids, directory, _ = campaign
    rows = [response(cid, 400 if i == 0 else 200) for i, cid in enumerate(ids)]
    rows[1]['response']['body']['choices'][0]['message']['content'] = 'NOT JSON'
    client.complete('original', rows)
    result = repair.advance(1, 'minimal', submit=True)
    assert result['status'] == 'requires_attention_nonserver_failure'
    assert result['blocked_nonserver_ids'] == ids[:1]
    assert not client.created


def test_cached_response_tampering_is_detected(campaign):
    client, ids, directory, _ = campaign
    client.complete('original', [response(cid, 500 if i == 0 else 200) for i, cid in enumerate(ids)])
    repair.advance(1, 'minimal')
    cached = directory/'r1-minimal/raw/original/output.jsonl'
    cached.write_bytes(cached.read_bytes().replace(b'[]', b'[ ]'))
    with pytest.raises(ValueError, match='Cached remote evidence'):
        repair.advance(1, 'minimal', submit=True)
    assert not client.created


def test_capacity_rejection_waits_for_new_release_and_retains_attempt(campaign):
    client, ids, directory, _ = campaign
    client.complete('original', [response(cid, 500 if i == 0 else 200) for i, cid in enumerate(ids)])
    repair.advance(1, 'minimal', submit=True)
    client.jobs['repair-1'].update(status='failed', errors={'data': [{'code': 'token_limit_exceeded'}]},
        request_counts={'total': 0, 'completed': 0, 'failed': 0})
    client.jobs['busy'] = {'id': 'busy', 'status': 'in_progress', 'input_file_id': 'unrelated',
        'metadata': {'model': repair.MODEL}}
    assert repair.advance(1, 'minimal', submit=True)['status'] == 'await_repair_capacity'
    assert len(client.created) == 1
    client.jobs['old_release'] = {'id': 'old_release', 'status': 'completed', 'completed_at': 50,
        'input_file_id': 'old-file', 'metadata': {'model': repair.MODEL}}
    assert repair.advance(1, 'minimal', submit=True, release_id='old_release')['status'] == 'await_repair_capacity'
    client.jobs['new_release'] = {'id': 'new_release', 'status': 'completed', 'completed_at': 101,
        'input_file_id': 'new-file', 'metadata': {'model': repair.MODEL}}
    result = repair.advance(1, 'minimal', submit=True, release_id='new_release')
    assert result['retry_count'] == 1 and len(client.created) == 2
    manifest = repair.batch.load_manifest(Path(result['repair_manifest']))
    assert manifest['capacity_retry_of'] == 'repair-1'
    assert manifest['capacity_release_trigger'] == 'new_release'
    assert len(list((directory/'r1-minimal/attempts').glob('*/manifest.json'))) == 2


def test_successful_api_response_with_parse_failure_blocks_final_comparison(campaign, monkeypatch):
    client, ids, directory, _ = campaign
    rows = [response(cid) for cid in ids]
    rows[0]['response']['body']['choices'][0]['message']['content'] = 'NOT JSON'
    client.complete('original', rows)
    def collect(args):
        m = repair.batch.load_manifest(Path(args.manifest))
        m.update(collected_at='now', failure_count=1)
        repair.save(args.manifest, m)
    monkeypatch.setattr(repair.batch, 'collect', collect)
    result = repair.advance(1, 'minimal', submit=True)
    assert result['status'] == 'requires_attention_parse_failure'
    assert not client.created


def credit_response(cid, status=429, error_type='insufficient_quota', code='credit_balance_exhausted'):
    row = response(cid, status)
    row['response']['body'] = {'error': {'type': error_type, 'code': code}}
    return row


def authorize_credit(directory, **changes):
    policy = repair.batch.load_manifest(directory/'authorization.json')
    policy.update(credit_balance_retries_authorized=True, credits_replenished_confirmed=True,
                  credit_balance_recovery_source_batch_ids=['original'])
    policy.update(changes)
    repair.save(directory/'authorization.json', policy)
    return policy


@pytest.mark.parametrize('status,error_type,code,eligible', [
    (429, 'insufficient_quota', 'credit_balance_exhausted', True),
    (429, 'rate_limit_exceeded', 'rate_limit_exceeded', False),
    (429, 'insufficient_quota', 'billing_hard_limit_reached', False),
    (429, 'insufficient_quota', None, False),
    (400, 'insufficient_quota', 'credit_balance_exhausted', False),
    (200, 'insufficient_quota', 'credit_balance_exhausted', False),
    (None, 'insufficient_quota', 'credit_balance_exhausted', False),
])
def test_credit_exhaustion_is_narrow_and_not_a_server_error(status, error_type, code, eligible):
    row = credit_response('x', status, error_type, code)
    assert repair.credit_balance_failure(row) is eligible
    assert not repair.server_failure(row)
    assert repair.retry_reason(row, {'server_failure_retries_authorized': True}) is None
    assert bool(repair.retry_reason(row, {'credit_balance_retries_authorized': True,
                                         'credits_replenished_confirmed': True})) is eligible


@pytest.mark.parametrize('authorization,credits', [(False, True), (True, False), (False, False)])
def test_credit_repair_requires_both_direct_authorization_and_replenishment(campaign, authorization, credits):
    client, ids, directory, _ = campaign
    authorize_credit(directory, credit_balance_retries_authorized=authorization,
                     credits_replenished_confirmed=credits)
    client.complete('original', [credit_response(ids[0])] + [response(cid) for cid in ids[1:]])
    result = repair.advance(1, 'minimal', submit=True)
    assert result['status'] == 'requires_attention_nonserver_failure'
    assert not client.created


def test_429_repair_is_exact_and_first_success_wins_across_later_server_repair(campaign):
    client, ids, directory, manifest_path = campaign
    original_manifest = manifest_path.read_bytes()
    authorize_credit(directory)
    rows = [credit_response(cid) if i < 2 else response(cid) for i, cid in enumerate(ids)]
    client.complete('original', rows)
    first = repair.advance(1, 'minimal', submit=True)
    assert first['eligible_credit_balance_failures'] == 2
    assert first['eligible_server_failures'] == 0
    manifest = repair.batch.load_manifest(Path(first['repair_manifest']))
    assert set(manifest['retry_reasons'].values()) == {'credit_balance_exhausted_429'}
    assert manifest['success_ids_excluded'] == sorted(ids[2:])
    lines = repair.input_lines(repair.batch.load_manifest(manifest_path))
    assert Path(manifest['input_jsonl']).read_bytes() == lines[ids[0]] + lines[ids[1]]
    for row in map(json.loads, Path(manifest['input_jsonl']).read_bytes().splitlines()):
        assert set(row['body']) == {'model', 'messages'}
        assert row['body']['model'] == repair.SNAPSHOT
    client.complete('repair-1', [response(ids[0]), response(ids[1], 500)])
    second = repair.advance(1, 'minimal', submit=True)
    assert second['eligible_credit_balance_failures'] == 0 and second['retry_count'] == 1
    assert client.created == [ids[:2], [ids[1]]]
    client.complete('repair-2', [response(ids[1])])
    final = repair.advance(1, 'minimal', submit=True)
    assert final['status'] == 'complete' and final['successes'] == 420
    provenance = repair.batch.load_manifest(directory/'r1-minimal/merged/provenance.json')['samples']
    assert provenance[ids[0]]['batch_id'] == 'repair-1'
    assert provenance[ids[1]]['batch_id'] == 'repair-2'
    assert all(provenance[cid]['batch_id'] == 'original' for cid in ids[2:])
    assert manifest_path.read_bytes() == original_manifest


def test_429_lost_submission_response_recovers_without_second_upload_or_batch(campaign):
    client, ids, directory, _ = campaign
    authorize_credit(directory)
    client.complete('original', [credit_response(ids[0])] + [response(cid) for cid in ids[1:]])
    client.lose_create_response = True
    with pytest.raises(TimeoutError):
        repair.advance(1, 'minimal', submit=True)
    files_before = set(client.contents)
    repair.advance(1, 'minimal', submit=True)
    assert client.created == [[ids[0]]] and set(client.contents) == files_before


def test_hold_prevents_any_upload_or_creation(campaign):
    client, ids, directory, _ = campaign
    authorize_credit(directory)
    client.complete('original', [credit_response(ids[0])] + [response(cid) for cid in ids[1:]])
    repair.save(directory.parent/'submission_hold.json', {'active': True})
    files_before = set(client.contents)
    with pytest.raises(ValueError, match='hold is active'):
        repair.advance(1, 'minimal', submit=True)
    assert not client.created and set(client.contents) == files_before


def test_renewed_exhaustion_stops_retries_until_new_replenishment_confirmation(campaign):
    client, ids, directory, _ = campaign
    authorize_credit(directory)
    client.complete('original', [credit_response(ids[0])] + [response(cid) for cid in ids[1:]])
    repair.advance(1, 'minimal', submit=True)
    client.complete('repair-1', [credit_response(ids[0])])
    result = repair.advance(1, 'minimal', submit=True)
    assert result['status'] == 'requires_attention_renewed_credit_exhaustion'
    assert repair.batch.load_manifest(directory.parent/'submission_hold.json')['active']
    assert client.created == [[ids[0]]]


@pytest.mark.parametrize('changes', [
    {'campaign': 'old_experiment'}, {'model_snapshot': 'other'}, {'rounds': [4]},
    {'conditions': ['full']}, {'script_sha256': 'different'}])
def test_authorization_scope_and_validated_script_hash_are_required(campaign, changes):
    client, ids, directory, _ = campaign
    authorize_credit(directory, **changes)
    client.complete('original', [credit_response(ids[0])] + [response(cid) for cid in ids[1:]])
    with pytest.raises(ValueError, match='scope mismatch|script hash'):
        repair.advance(1, 'minimal', submit=True)
    assert not client.created
