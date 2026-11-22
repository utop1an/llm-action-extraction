import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.risec_spans import make_span_examples


def test_feature_preparation_needs_no_test_file(tmp_path):
    data, cache, out = [tmp_path / x for x in ('data', 'cache', 'features')]
    data.mkdir(); cache.mkdir()
    for split in ('train', 'dev'):
        row = {'id': split, 'text': 'Mix rice.', 'words': ['Mix','rice','.'], 'token_offsets': [[0,3],[4,8],[8,9]],
               'actions': [{'trigger_tokens':[0], 'text':'Mix', 'explicit_arguments':[{'token_indices':[1], 'target_text':'rice'}]}],
               'argument_links_available':True}
        (data/(split+'.jsonl')).write_text(json.dumps(row)+'\n', encoding='utf-8')
        for mode in ('act', 'arg'):
            for ex in make_span_examples([row], mode):
                np.save(cache/(ex['embedding_key']+'.npy'),np.zeros((3,868),dtype=np.float32))
    subprocess.run([sys.executable,'scripts/prepare_ceasdrl_holdout.py','--data',str(data),'--output',str(out),
                    '--existing-features',str(cache)],check=True,capture_output=True)
    manifest=json.loads((out/'manifest.json').read_text())
    assert set(manifest['dataset_sha256']) == {'train','dev'}
    assert all(splits['test']==[] for splits in manifest['examples'].values())
    assert manifest['new_embedding_contexts']==0


def test_trainer_rejects_test_features_and_document_leakage():
    sys.path.insert(0,str(Path('scripts').resolve()))
    from train_ceasdrl_holdout import validate_holdout
    f=SimpleNamespace(examples={'train':[{'doc_id':'a'}],'dev':[{'doc_id':'b'}],'test':[]},
                      manifest={'dataset_sha256':{'train':'x','dev':'y'}})
    validate_holdout(f,True)
    f.examples['test']=[{'doc_id':'c'}]
    with pytest.raises(ValueError,match='empty test'):validate_holdout(f,True)
    f.examples['test']=[];f.examples['dev']=[{'doc_id':'a'}]
    with pytest.raises(ValueError,match='leakage'):validate_holdout(f,True)
