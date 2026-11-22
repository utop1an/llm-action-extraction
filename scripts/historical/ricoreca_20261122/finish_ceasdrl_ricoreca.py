"""Finish validation/export/evaluation after the two already-running trainers complete."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path('results/ceasdrl_ricoreca_64_16_20_seed0'))
    p.add_argument('--data-root', type=Path, default=Path('data/procedural/ricoreca'))
    p.add_argument('--legacy-python', required=True)
    args = p.parse_args()
    status_path = args.run / 'finalization_status.json'
    log_path = args.run / 'finalization.log'
    if status_path.exists():
        raise ValueError('Finalizer already started; inspect existing status before retrying')
    def status(phase, **extra):
        obj = dict(phase=phase, updated_at_utc=datetime.now(timezone.utc).isoformat(), **extra)
        temp = status_path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
        temp.replace(status_path)
    status('waiting_for_training')
    try:
        start = time.time()
        while not all((args.run / m / 'receipt.json').exists() for m in ('act', 'arg')):
            if time.time()-start > 4*3600:
                raise TimeoutError('Training did not finish within four hours; no test evaluation performed')
            time.sleep(15)
        for m in ('act', 'arg'):
            r = json.loads((args.run / m / 'receipt.json').read_text(encoding='utf-8'))
            if r['status'] != 'completed' or not all(h['complete'] for h in r['history']):
                raise ValueError('Training is not a completed full-epoch run')
        env = dict(os.environ)
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONWARNINGS'] = 'ignore::FutureWarning'
        with log_path.open('w', encoding='utf-8') as log:
            def run(phase, executable, script, parameters):
                status(phase)
                command = [executable, 'scripts/ceasdrl/'+script] + [str(x) for x in parameters]
                log.write(json.dumps(command)+'\n');log.flush()
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=env)
            for m in ('act', 'arg'):
                run('export_'+m, sys.executable, 'export_ceasdrl_holdout_checkpoint.py',
                    ['--run', args.run/m, '--features', args.data_root/'features_train_dev'])
                run('keras_compatibility_'+m, args.legacy_python, 'check_ceasdrl_holdout_keras.py', ['--run',args.run/m])
            run('sealed_test_evaluation',sys.executable,'evaluate_ceasdrl_holdout.py',
                ['--data',args.data_root/'converted','--features',args.data_root/'features_train_dev',
                 '--action-run',args.run/'act','--argument-run',args.run/'arg',
                 '--legacy-python',args.legacy_python,'--output',args.run/'final'])
            run('independent_audit',sys.executable,'audit_ceasdrl_ricoreca.py',['--data-root',args.data_root,'--run',args.run])
            run('report',sys.executable,'summarize_ceasdrl_ricoreca.py',['--run',args.run])
        receipt = json.loads((args.run/'final/receipt.json').read_text(encoding='utf-8'))
        path = args.data_root/'experiment_64_16_20.json'
        config = json.loads(path.read_text(encoding='utf-8'))
        config.update(status='completed', scores=receipt['metrics'], remaining_prerequisites=[],
                      result_report=(args.run/'REPORT.md').as_posix())
        path.write_text(json.dumps(config,indent=2),encoding='utf-8')
        status('completed', metrics=receipt['metrics'])
        print(json.dumps({'status':'completed','metrics':receipt['metrics']}),flush=True)
    except Exception as error:
        status('failed',error=str(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    main()
