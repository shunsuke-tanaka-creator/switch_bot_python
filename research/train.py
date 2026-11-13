"""Chronological, day-disjoint evaluation. Test data never selects a checkpoint."""
from pathlib import Path
import copy
import fcntl
import hashlib
import json
import os
import random
import numpy as np
import torch
from .data import F, BASE, FEATURES, VERSION, SCALES, Timeline, materialize
from .evaluation import online_updates
from .model import HierarchicalLSTM


from .io import atomic_json
from .provenance import run_metadata


def batch(samples, device, mean, std):
    samples = [materialize(s) for s in samples]
    out={k:torch.as_tensor(np.stack([s[k] for s in samples]),device=device) for k in samples[0]}
    for k in ['x','history']:
        raw=out[k].float(); scaled=(raw-mean)/std
        # Normalize only observed measurements. Presence bits and time features retain their meaning.
        n=len(BASE); scaled[...,:n]*=raw[...,n:2*n]
        if k=='x': scaled*= (torch.arange(12,device=device)[None,None,None,:]<out['lengths'][...,None])[...,None]
        out[k]=scaled
    return out


def normalization(examples):
    # Streaming parallel-variance aggregation: never concatenate all training tensors.
    n = len(BASE)
    counts = np.zeros(n, np.int64)
    averages = np.zeros(n, np.float64)
    m2 = np.zeros(n, np.float64)
    for _, spec in examples:
        sample = materialize(spec)
        x = sample['x'][np.arange(12)[None,None,:] < sample['lengths'][...,None]]
        for j in range(n):
            values = x[:,j][x[:,j+n] > 0].astype(np.float64)
            if not len(values):
                continue
            count = len(values)
            avg = values.mean()
            delta = avg - averages[j]
            total = counts[j] + count
            m2[j] += ((values-avg)**2).sum() + delta**2 * counts[j] * count / total
            averages[j] += delta * count / total
            counts[j] = total
    mean = np.zeros(F, np.float32)
    std = np.ones(F, np.float32)
    mean[:n] = averages
    std[:n] = np.where(counts > 0, np.maximum(np.sqrt(m2/np.maximum(counts,1)), .1), 1)
    return mean, std


def metrics(model, examples, device, mean, std):
    sums={scope:np.zeros(5) for scope in ('current','daily')};counts={s:np.zeros(5,dtype=int) for s in sums}
    correct=total=0
    model.eval()
    with torch.no_grad():
        for l,s in examples:
            pred=model(batch([s],device,mean,std))
            if l.scope in sums:
                p=pred[l.scope][0].cpu().numpy();mask=np.isfinite(l.values)
                sums[l.scope][mask]+=np.abs(p[mask]-l.values[mask]);counts[l.scope]+=mask
            elif l.scope=='activity':
                correct+=int(pred['activity'][0].argmax().item()==l.activity);total+=1
    result={s:{k:dict(n=int(counts[s][i]),mae=float(sums[s][i]/counts[s][i]) if counts[s][i] else None) for i,k in enumerate(SCALES)} for s in sums}
    values=[v['mae'] for s in sums for v in result[s].values() if v['mae'] is not None]
    result.update(mean_mae=float(np.mean(values)) if values else None,activity_accuracy=correct/total if total else None,activity_n=total)
    return result


def train(rows, directory, variant='full', use_text=True, adaptation=True, epochs=20, hidden=48, seed=17, evaluate_test=False, min_labels=20, text_model=None,
          data_mode='real', contains_sample=False, source_fingerprint=None, source_max_seq=None, sample_manifest_hash=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'training.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return dict(status='busy',data_mode=data_mode,contains_sample=contains_sample,
                                          fingerprint=source_fingerprint,source_max_seq=source_max_seq or {},
                                          sample_manifest_hash=sample_manifest_hash)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        torch.set_num_threads(min(4,os.cpu_count() or 1))
        timeline=Timeline(rows,text_model=text_model)
        examples=timeline.examples()
        state=[e for e in examples if e[0].scope in ('current','daily')]
        dates=sorted({l.at.date() for l,_ in state})
        if len(state)<min_labels or len(dates)<5:
            result=dict(status='insufficient_labels',state_answers=len(state),days=len(dates),required_answers=min_labels,required_days=5,
                        data_mode=data_mode,contains_sample=contains_sample,fingerprint=source_fingerprint,
                        source_max_seq=source_max_seq or {},sample_manifest_hash=sample_manifest_hash)
            atomic_json(directory/'status.json',result);return result
        a=max(1,int(len(dates)*0.6));b=max(a+1,int(len(dates)*0.8))
        train_end=dates[a];test_start=dates[min(b,len(dates)-1)]
        # Labels answered later than a split boundary cannot train an earlier model.
        tr=[e for e in examples if e[0].at.date()<train_end and e[0].available.date()<train_end]
        va=[e for e in examples if train_end<=e[0].at.date()<test_start and e[0].available.date()<test_start]
        te=[e for e in examples if e[0].at.date()>=test_start]
        if not tr or not va:
            result=dict(status='insufficient_split',data_mode=data_mode,contains_sample=contains_sample,fingerprint=source_fingerprint,
                        source_max_seq=source_max_seq or {},sample_manifest_hash=sample_manifest_hash)
            atomic_json(directory/'status.json',result);return result
        mean_np,std_np=normalization(tr)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mean=torch.tensor(mean_np,device=device);std=torch.tensor(std_np,device=device)
        model=HierarchicalLSTM(hidden,variant,use_text,adaptation).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.001)
        best=None;best_mae=float('inf')
        for epoch in range(epochs):
            model.train();order=list(range(len(tr)));random.shuffle(order)
            for offset in range(0,len(order),4):
                chunk=[tr[i] for i in order[offset:offset+4]]
                outputs=model(batch([s for _,s in chunk],device,mean,std));losses=[]
                for i,(l,_) in enumerate(chunk):
                    if l.scope=='activity': losses.append(.2*torch.nn.functional.cross_entropy(outputs['activity'][i:i+1],torch.tensor([l.activity],device=device)))
                    else:
                        mask=np.isfinite(l.values)
                        if mask.any():
                            target=torch.tensor(np.nan_to_num(l.values,nan=3.),device=device)
                            losses.append(torch.nn.functional.smooth_l1_loss(outputs[l.scope][i][mask],target[mask]))
                if not losses:continue
                opt.zero_grad();loss=torch.stack(losses).mean()
                if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);opt.step()
            valid=metrics(model,va,device,mean,std)
            score=valid['mean_mae']
            if score is not None and score<best_mae:
                best_mae=score;best=copy.deepcopy(model.state_dict())
        if best is None:
            result=dict(status='insufficient_validation_labels',data_mode=data_mode,contains_sample=contains_sample,fingerprint=source_fingerprint,
                        source_max_seq=source_max_seq or {},sample_manifest_hash=sample_manifest_hash)
            atomic_json(directory/'status.json',result);return result
        model.load_state_dict(best)
        support={scope:[int(sum(l.scope==scope and np.isfinite(l.values[i]) for l,_ in tr)) for i in range(5)] for scope in ('current','daily')}
        identity=hashlib.sha256(json.dumps([(l.id,l.scope,l.at.isoformat(),[None if not np.isfinite(v) else float(v) for v in l.values]) for l,_ in examples],sort_keys=True).encode()).hexdigest()
        report=dict(status='trained',version=VERSION,fingerprint=source_fingerprint or identity,label_fingerprint=identity,
                    data_mode=data_mode,contains_sample=contains_sample,source_max_seq=source_max_seq or {},
                    sample_manifest_hash=sample_manifest_hash,device=str(device),settings=model.settings,
                    train_n=len(tr),validation_n=len(va),test_n=len(te),train_before=str(train_end),test_from=str(test_start),
                    support=support,activity_train_n=sum(l.scope=='activity' for l,_ in tr),text_model=text_model,text_identity=timeline.encoder.identity if timeline.encoder else 'character-ngram-v1',validation=metrics(model,va,device,mean,std),test=metrics(model,te,device,mean,std) if evaluate_test else None,
                    limitations=['Research estimate, not a diagnosis','Door opening does not establish occupancy',('Frozen local sentence encoder' if text_model else 'Character n-gram text features; no pretrained language encoder'),'Historical reception times are not known for legacy imports'])
        report['run'] = run_metadata(seed, epochs, dict(model.settings, optimizer='AdamW', lr=.001, weight_decay=.001, batch_size=4, feature_version=VERSION),
                                    {name:[dict(id=l.id,at=l.at.isoformat(),available=l.available.isoformat()) for l,_ in items] for name,items in [('train',tr),('validation',va),('test',te)]})
        atomic_json(directory/'run.json', report['run'])
        checkpoint=dict(weights=best,settings=model.settings,mean=mean_np.tolist(),std=std_np.tolist(),report=report,features=FEATURES,version=VERSION)
        tmp=directory/'candidate.tmp';torch.save(checkpoint,tmp);os.replace(tmp,directory/'candidate.pt')
        # New model promoted only using validation, never test scores. Old model remains on failure.
        active=directory/'active.pt';promote=True
        if active.exists():
            old=torch.load(active,map_location=device,weights_only=True)
            if old['settings']==model.settings and old.get('version')==VERSION and old['report'].get('text_identity')==report['text_identity']:
                prior=HierarchicalLSTM(**old['settings']).to(device);prior.load_state_dict(old['weights'])
                prior_score=metrics(prior,va,device,torch.tensor(old['mean'],device=device),torch.tensor(old['std'],device=device))['mean_mae']
                promote=prior_score is None or best_mae<=prior_score
        if promote:
            if active.exists():
                import shutil
                shutil.copy2(active,directory/'previous.pt')
            tmp=directory/'active.tmp';torch.save(checkpoint,tmp);os.replace(tmp,active)
        report['promoted']=promote;atomic_json(directory/'status.json',report)
        return report


def ablations(rows,directory,**kwargs):
    result={}
    for name,settings in [('micro',dict(variant='micro')),('micro_meso',dict(variant='micro_meso')),
                          ('full',{}),('no_adaptation',dict(adaptation=False)),('no_text',dict(use_text=False))]:
        result[name]=train(rows,Path(directory)/name,evaluate_test=True,**settings,**kwargs)
    atomic_json(Path(directory)/'comparison.json',result)
    return result


def personalization_evaluation(rows, directory, epochs=20, hidden=48, text_model=None, data_mode='real', contains_sample=False,
                               source_fingerprint=None, source_max_seq=None, sample_manifest_hash=None):
    """Frozen weights vs event-wise adaptation. Only earlier received answers can update weights."""
    base=Path(directory)/'frozen_base'
    report=train(rows,base,epochs=epochs,hidden=hidden,text_model=text_model,data_mode=data_mode,contains_sample=contains_sample,
                 source_fingerprint=source_fingerprint,source_max_seq=source_max_seq,sample_manifest_hash=sample_manifest_hash)
    if report['status']!='trained':return report
    state=torch.load(base/'candidate.pt',map_location='cpu',weights_only=True)
    frozen=HierarchicalLSTM(**state['settings']);frozen.load_state_dict(state['weights']);frozen.eval()
    adapted=copy.deepcopy(frozen)
    optimizer=torch.optim.AdamW(adapted.parameters(),lr=.0001)
    examples=Timeline(rows,text_model=text_model).examples()
    test=[e for e in examples if str(e[0].at.date())>=report['test_from'] and e[0].scope in ('current','daily')]
    mean=torch.tensor(state['mean']);std=torch.tensor(state['std'])
    totals={name:{scope:np.zeros(5) for scope in ('current','daily')} for name in ('frozen','updated')}
    counts={scope:np.zeros(5,dtype=int) for scope in ('current','daily')};predictions=[]
    used=set()
    for target_label, sample in sorted(test, key=lambda e:(e[0].at,e[0].id)):
        new=online_updates(examples,used,target_label.at,report['train_before'])
        adapted.train()
        for label, prior_sample in new:
            out=adapted(batch([prior_sample],'cpu',mean,std))
            if label.scope=='activity':
                loss=torch.nn.functional.cross_entropy(out['activity'],torch.tensor([label.activity]))
            else:
                mask=np.isfinite(label.values)
                if not mask.any(): continue
                loss=torch.nn.functional.smooth_l1_loss(out[label.scope][0][mask],torch.tensor(label.values[mask]))
            if not torch.isfinite(loss): raise ValueError('Non-finite adaptation loss')
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted.parameters(),1,error_if_nonfinite=True)
            optimizer.step();used.add(label.id)
        adapted.eval()
        b=batch([sample],'cpu',mean,std)
        mask=np.isfinite(target_label.values);counts[target_label.scope]+=mask
        item=dict(id=target_label.id,at=target_label.at.isoformat(),scope=target_label.scope,
                  updated_on_labels=len(used))
        with torch.no_grad():
            for name,model in [('frozen',frozen),('updated',adapted)]:
                values=model(b)[target_label.scope][0].numpy()
                totals[name][target_label.scope][mask]+=np.abs(values[mask]-target_label.values[mask])
                item[name]=values.tolist()
        predictions.append(item)
    result=dict(protocol='prequential-event: predict before receiving the target answer; earlier same-day answers allowed',test_from=report['test_from'],
                data_mode=data_mode,contains_sample=contains_sample,fingerprint=source_fingerprint,
                source_max_seq=source_max_seq or {},sample_manifest_hash=sample_manifest_hash,
                updated_on_labels=len(used),predictions=predictions,metrics={})
    for name in totals:
        result['metrics'][name]={scope:{k:dict(n=int(counts[scope][i]),mae=float(totals[name][scope][i]/counts[scope][i]) if counts[scope][i] else None) for i,k in enumerate(SCALES)} for scope in counts}
    atomic_json(Path(directory)/'personalization.json',result)
    return result
