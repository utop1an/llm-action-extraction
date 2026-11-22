"""Train the six cEASDRL stages of the frozen two-source, one-target experiment."""
import argparse
import copy
import hashlib
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
os.environ.setdefault('TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT','64')
os.environ.setdefault('CUDNN_CONV_WSCAP_DBG','128')
import torch
from ceasdrl_torch import CnnDqn, Replay, dqn_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import evaluation
from src.evaluation_helpers import match_objs

SPLIT = ROOT / 'data/easdrl/indices/ceasdrl_two_source_v1.json'
WIDTH = {'act': 1000, 'arg': 200}


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    tmp.replace(path)


def load_docs(manifest):
    docs = {}
    for d, spec in manifest['domains'].items():
        raw = (ROOT / spec['source']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == spec['sha256']
        rows = pickle.loads(raw, encoding='latin1')
        ids = spec['doc_ids']
        assert sorted(ids['train'] + ids['val'] + ids['test']) == list(range(len(rows)))
        docs.update({(d, i): row for i, row in enumerate(rows)})
    return docs


class Examples:
    def __init__(self, docs, cache):
        self.docs, self.cache, self.matrices = docs, cache, {}

    def make(self, key, head=None, annotation=None):
        doc = self.docs[key]
        if head is None:
            offset, field, words = 0, 'act', doc['words']
        else:
            sent = doc['word2sent'][head]
            offset = sum(len(s) for s in doc['sents'][:max(0, sent-1)])
            field = 'arg_' + str(sent)
            words = sum(doc['sents'][max(0, sent-1):sent+1], [])
        kind = 'act' if head is None else 'arg'
        if len(words) > WIDTH[kind]:
            raise ValueError('Context exceeds declared capacity: ' + str(key))
        labels = {}
        relations = {}
        if annotation is not None and kind == 'act':
            for a in annotation:
                i = a['act_idx']
                labels[i] = a['act_type'] + 1
                relations[i] = a['related_acts']
        elif annotation is not None:
            left, right = [[i-offset for i in side if offset <= i < offset+len(words)]
                           for side in annotation['obj_idxs']]
            for group, other in [(left, right), (right, left)]:
                for i in group:
                    labels[i] = 4 if annotation['obj_idxs'][1] else 2
                    relations[i] = other
        return dict(key=key, field=field, words=words, offset=offset, head=head,
                    labels=labels, relations=relations, annotation=annotation, kind=kind)

    def batch(self, items):
        kind = items[0]['kind']
        # With zero padding, five extra zero positions retain every possible
        # constant-padding convolution response for kernels of width 2..5.
        needed = max(len(x['words']) for x in items)+5
        width = min(WIDTH[kind], ((needed+31)//32)*32)
        words = np.zeros((len(items), width, 868), dtype=np.float32)
        dist = np.zeros((len(items), width), dtype=np.float32)
        for j,x in enumerate(items):
            k = x['key']
            if k not in self.matrices:
                path = self.cache / ('%s_%d.npz' % k)
                with np.load(path) as data:
                    self.matrices[k] = {f:data[f] for f in data.files}
            n = len(x['words'])
            mat = self.matrices[k][x['field']]
            assert mat.shape == (n,868)
            words[j,:n] = mat
            if kind == 'arg':
                dist[j,:n] = np.abs(np.arange(n) - (x['head']-x['offset']))
        return torch.from_numpy(words).cuda(), torch.from_numpy(dist).cuda()


def reward(ex, i, decision, tags, count, terminal):
    # Preserve upstream binary-decision semantics, including the arg EX branch
    # in cEnvironment.act, whose action==2 cases cannot fire for 0/1 decisions.
    label = ex['labels'].get(i, 1)
    if label == 4:
        related = ex['relations'][i]
        taken = any(tags[k] == 2 for k in related)
        good = decision == 1 and not taken
        if ex['kind'] == 'act' and decision == 0:
            good = taken or any(k > i for k in related)
        value = 150 if good else -150
    elif label == 3:
        value = 50 if decision else 0
    elif label == 2:
        value = 100 if decision else -100
    else:
        value = -50 if decision else 50
    rate = count / WIDTH[ex['kind']]
    if value:
        value += (1 if value > 0 and rate <= .1 else -1) * 250 * rate**2
    return value + 2 * terminal


@torch.no_grad()
def predict(model, data, examples):
    model.eval()
    outputs = []
    for start in range(0,len(examples),24):
        items = examples[start:start+24]
        w,d = data.batch(items)
        parts = model.static_parts(w,d)
        tags = torch.zeros(w.shape[:2],device=w.device)
        lengths = torch.tensor([len(x['words']) for x in items],device=w.device)
        for i in range(int(lengths.max())):
            tags[:,i] = (model.from_static(parts,tags).argmax(1)+1)*(i<lengths)
        outputs.extend([np.flatnonzero(row==2).tolist() for row in tags.cpu().numpy()])
    return outputs


def dev_score(mode, data, items, predictions):
    if mode == 'act':
        rows = [dict(data.docs[x['key']],pred=[dict(verb=x['words'][i], arguments=[]) for i in p])
                for x,p in zip(items,predictions)]
        return evaluation(rows)[2]
    counts = np.zeros(3)
    for x,p in zip(items,predictions):
        doc = data.docs[x['key']]
        gold = [[doc['words'][i] for i in side if i>=0] for side in x['annotation']['obj_idxs']]
        args = [x['words'][i] for i in p if i+x['offset']!=x['head']]
        counts += match_objs(gold,args)[:3]
    return float(2*counts[0]/max(1,counts[1]+counts[2]))


def train(mode, data, train_keys, val_keys, path, seed, status):
    def examples(keys):
        if mode == 'act':
            return [data.make(k,annotation=data.docs[k]['acts']) for k in keys]
        return [data.make(k,a['act_idx'],a) for k in keys for a in data.docs[k]['acts']]
    train_set, dev_set = examples(train_keys), examples(val_keys)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = CnnDqn(mode).cuda()
    target = copy.deepcopy(model).eval()
    opt = torch.optim.Adam(model.parameters(),lr=.001)
    memory = Replay()
    steps = updates = ticks = stale = 0
    best = -1
    history = []
    decay = sum(len(x['words']) for x in train_set)
    for epoch in range(1,6):
        order = list(range(len(train_set)))
        random.shuffle(order)
        model.train()
        for start in range(0,len(order),16):
            ids = order[start:start+16]
            items = [train_set[k] for k in ids]
            w,d = data.batch(items)
            tags = np.zeros(w.shape[:2],dtype=np.uint8)
            lengths = np.array([len(x['words']) for x in items])
            counts = np.zeros(len(items),dtype=int)
            version = -1
            for i in range(int(lengths.max())):
                with torch.no_grad():
                    if version != updates:
                        parts = model.static_parts(w,d)
                        version = updates
                    choice = model.from_static(parts,torch.from_numpy(tags).cuda().float()).argmax(1).cpu().numpy()
                explore = np.random.random(len(items)) < max(.1,1-.9*steps/max(1000,decay))
                choice[explore] = np.random.randint(2,size=int(explore.sum()))
                active = np.flatnonzero(i<lengths)
                for j in active:
                    before = tags[j,:i].copy()
                    a = int(choice[j])
                    tags[j,i] = a+1
                    counts[j] += a
                    done = bool(i+1==lengths[j])
                    r = reward(items[j],i,a,tags[j],counts[j],done)
                    memory.add((ids[j],before,a,r,done))
                steps += len(active)
                ticks += 1
                if ticks%4==0 and len(memory.rows)>=32:
                    sample = memory.sample(32)
                    opt.zero_grad(set_to_none=True)
                    batch_loss = 0.
                    for sub_start in range(0,len(sample),8):
                        sub = sample[sub_start:sub_start+8]
                        bw,bd = data.batch([train_set[t[0]] for t in sub])
                        pre = np.zeros(bw.shape[:2],dtype=np.float32)
                        post = pre.copy()
                        for j,(_,old,a,_,_) in enumerate(sub):
                            pre[j,:len(old)] = post[j,:len(old)] = old
                            post[j,len(old)] = a+1
                        actions, rewards, terminals = zip(*(t[2:] for t in sub))
                        loss = dqn_loss(model,target,bw,torch.from_numpy(pre).cuda(),torch.from_numpy(post).cuda(),
                                        bd,torch.tensor(actions,device='cuda'),torch.tensor(rewards,dtype=torch.float32,device='cuda'),
                                        torch.tensor(terminals,device='cuda')) * (len(sub)/len(sample))
                        if not torch.isfinite(loss):
                            raise RuntimeError('Nonfinite training loss')
                        loss.backward()
                        batch_loss += loss.item()
                        del bw,bd,loss
                    opt.step()
                    updates += 1
                    if updates%100==0:
                        target.load_state_dict(model.state_dict())
                    if updates==1 or updates%100==0:
                        save(status,dict(target=path.name,stage=mode,epoch=epoch,updates=updates,steps=steps,loss=batch_loss))
        f1 = dev_score(mode,data,dev_set,predict(model,data,dev_set))
        history.append(dict(epoch=epoch,updates=updates,source_val_f1=f1))
        save(path/(mode+'_history.json'),history)
        if f1>best:
            best,stale=f1,0
            torch.save(model.state_dict(),path/(mode+'_best.pt'))
        else:
            stale+=1
        if stale>=3:
            break
    model.load_state_dict(torch.load(path/(mode+'_best.pt'),weights_only=True))
    return model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--features',type=Path)
    args=parser.parse_args()
    run=args.run
    status=run/'training_progress.json'
    manifest=json.loads(SPLIT.read_text())
    docs=load_docs(manifest)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    config=dict(protocol=manifest['protocol'],split_sha256=hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
                seed=0,epochs=5,patience=3,device=torch.cuda.get_device_name(0),width=WIDTH,
                replay_batch=32,gradient_microbatch=8,feature_cache=str(args.features or run/'features'),
                source_val_metric={'act':'pooled source Action F1','arg':'pooled source gold-action-conditioned Argument F1'},
                source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__),Path(__file__).with_name('prepare_ceasdrl_ood_embeddings.py'),
                     Path(__file__).with_name('ceasdrl_torch.py'),ROOT/'evaluation.py',ROOT/'src/evaluation_helpers.py']})
    if (run/'config.json').exists():
        raise FileExistsError('Existing training run')
    save(run/'config.json',config)
    for target in ['wikihow','cooking','win2k']:
        path=run/target
        path.mkdir()
        spec=manifest['target_runs'][target]
        def resolve(refs):
            result=[]
            for ref in refs:
                domain,split=ref.split('.')
                result.extend((domain,i) for i in manifest['domains'][domain]['doc_ids'][split])
            return result
        train_keys=resolve(spec['ceasdrl_ood']['train'])
        val_keys=resolve(spec['ceasdrl_ood']['val'])
        test_keys=resolve([spec['test_for_all_three_methods']])
        assert not set(train_keys)&set(val_keys) and all(d!=target for d,_ in train_keys+val_keys)
        assert all(d==target for d,_ in test_keys)
        save(path/'documents.json',dict(train=train_keys,val=val_keys,test=test_keys))
        data=Examples(docs,args.features or run/'features')
        def wait_for(keys):
            deadline=time.monotonic()+3600
            while not all((data.cache/('%s_%d.npz' % k)).is_file() for k in keys):
                if time.monotonic()>deadline:
                    raise TimeoutError('Feature preparation incomplete')
                time.sleep(3)
        save(status,dict(target=target,phase='waiting for source embeddings'))
        wait_for(train_keys+val_keys)
        action=train('act',data,train_keys,val_keys,path,0,status)
        argument=train('arg',data,train_keys,val_keys,path,0,status)
        wait_for(test_keys)
        heads=predict(action,data,[data.make(k) for k in test_keys])
        contexts=[data.make(k,i) for k,indices in zip(test_keys,heads) for i in indices]
        found=predict(argument,data,contexts)
        by_doc={k:[] for k in test_keys}
        predicted_actions={k:set(indices) for k,indices in zip(test_keys,heads)}
        for ex,indices in zip(contexts,found):
            k=ex['key']
            args_text=[ex['words'][i] for i in indices if i+ex['offset'] not in predicted_actions[k]]
            by_doc[k].append(dict(verb=docs[k]['words'][ex['head']],arguments=args_text))
        save(path/'test_predictions.json',[dict(dataset=k[0],doc_id=k[1],prediction=by_doc[k]) for k in test_keys])
        metrics=evaluation([dict(docs[k],pred=by_doc[k]) for k in test_keys])
        save(path/'metrics.json',dict(action_f1=metrics[2],strict_argument_f1=metrics[5],documents=len(test_keys)))
    save(status,dict(phase='completed',targets=list(manifest['domains'])))


if __name__=='__main__':
    main()
