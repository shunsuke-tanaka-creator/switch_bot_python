"""Chronological, day-disjoint evaluation. Test data never selects a checkpoint."""
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
import copy
import fcntl
import hashlib
import json
import os
import random
import numpy as np
import torch
from .data import F, BASE, TEXT_DIM, FEATURES, VERSION, SCALES, Timeline
from .model import HierarchicalLSTM


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));os.replace(tmp,path)


def batch(samples, device, mean, std):
    out={k:torch.as_tensor(np.stack([s[k] for s in samples]),device=device) for k in samples[0]}
    for k in ['x','history']:
        raw=out[k].float(); scaled=(raw-mean)/std
        # Normalize only observed measurements. Presence bits and time features retain their meaning.
        n=len(BASE); scaled[...,:n]*=raw[...,n:2*n]
        if k=='x': scaled*= (torch.arange(12,device=device)[None,None,None,:]<out['lengths'][...,None])[...,None]
        out[k]=scaled
    return out


def normalization(examples):
    # Fit only features from training windows; no validation/test input is consulted.
    x=np.concatenate([s['x'][np.arange(12)[None,None,:]<s['lengths'][...,None]] for _,s in examples])
    mean=np.zeros(F,np.float32);std=np.ones(F,np.float32)
    for j in range(len(BASE)):
        values=x[:,j][x[:,j+len(BASE)]>0]
        if len(values): mean[j]=values.mean();std[j]=max(float(values.std()),0.1)
    return mean,std


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


def train(rows, directory, variant='full', use_text=True, adaptation=True, epochs=20, hidden=48, seed=17, evaluate_test=False, min_labels=20, text_model=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'training.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return dict(status='busy')
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        torch.set_num_threads(min(4,os.cpu_count() or 1))
        timeline=Timeline(rows,text_model=text_model)
        examples=timeline.examples()
        state=[e for e in examples if e[0].scope in ('current','daily')]
        dates=sorted({l.at.date() for l,_ in state})
        if len(state)<min_labels or len(dates)<5:
            result=dict(status='insufficient_labels',state_answers=len(state),days=len(dates),required_answers=min_labels,required_days=5)
            atomic_json(directory/'status.json',result);return result
        a=max(1,int(len(dates)*0.6));b=max(a+1,int(len(dates)*0.8))
        train_end=dates[a];test_start=dates[min(b,len(dates)-1)]
        # Labels answered later than a split boundary cannot train an earlier model.
        tr=[e for e in examples if e[0].at.date()<train_end and e[0].available.date()<train_end]
        va=[e for e in examples if train_end<=e[0].at.date()<test_start and e[0].available.date()<test_start]
        te=[e for e in examples if e[0].at.date()>=test_start]
        if not tr or not va:return dict(status='insufficient_split')
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
                opt.zero_grad();torch.stack(losses).mean().backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            valid=metrics(model,va,device,mean,std)
            score=valid['mean_mae']
            if score is not None and score<best_mae:
                best_mae=score;best=copy.deepcopy(model.state_dict())
        if best is None:return dict(status='insufficient_validation_labels')
        model.load_state_dict(best)
        support={scope:[int(sum(l.scope==scope and np.isfinite(l.values[i]) for l,_ in tr)) for i in range(5)] for scope in ('current','daily')}
        identity=hashlib.sha256(json.dumps([(l.id,l.scope,l.at.isoformat(),[None if not np.isfinite(v) else float(v) for v in l.values]) for l,_ in examples],sort_keys=True).encode()).hexdigest()
        report=dict(status='trained',version=VERSION,fingerprint=identity,device=str(device),settings=model.settings,
                    train_n=len(tr),validation_n=len(va),test_n=len(te),train_before=str(train_end),test_from=str(test_start),
                    support=support,activity_train_n=sum(l.scope=='activity' for l,_ in tr),text_model=text_model,text_identity=timeline.encoder.identity if timeline.encoder else 'character-ngram-v1',validation=metrics(model,va,device,mean,std),test=metrics(model,te,device,mean,std) if evaluate_test else None,
                    limitations=['Research estimate, not a diagnosis','Door opening does not establish occupancy','Character n-gram text features; no pretrained language encoder','Historical reception times are not known for legacy imports'])
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


def personalization_evaluation(rows, directory, epochs=20, hidden=48, text_model=None):
    """Frozen personal model vs sequential fine tuning. Predict a held-out day BEFORE using its labels."""
    base=Path(directory)/'frozen_base'
    report=train(rows,base,epochs=epochs,hidden=hidden,text_model=text_model)
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
    days=sorted({l.at.date() for l,_ in test})
    # Only labels from days strictly before the day under evaluation can update weights.
    available=[];used=set()
    for day in days:
        new=[e for e in examples if e[0].id not in used and e[0].at.date()>=datetime.fromisoformat(report['train_before']).date() and e[0].available.date()<day]
        adapted.train()
        for l,s in new:
            out=adapted(batch([s],'cpu',mean,std))
            if l.scope=='activity':loss=torch.nn.functional.cross_entropy(out['activity'],torch.tensor([l.activity]))
            else:
                mask=np.isfinite(l.values)
                if not mask.any():continue
                loss=torch.nn.functional.smooth_l1_loss(out[l.scope][0][mask],torch.tensor(l.values[mask]))
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(adapted.parameters(),1);optimizer.step();used.add(l.id)
        adapted.eval()
        for l,s in [e for e in test if e[0].at.date()==day]:
            b=batch([s],'cpu',mean,std);mask=np.isfinite(l.values);counts[l.scope]+=mask
            item=dict(id=l.id,at=l.at.isoformat(),scope=l.scope)
            with torch.no_grad():
                for name,m in [('frozen',frozen),('updated',adapted)]:
                    values=m(b)[l.scope][0].numpy();totals[name][l.scope][mask]+=np.abs(values[mask]-l.values[mask]);item[name]=values.tolist()
            predictions.append(item)
    result=dict(protocol='prequential: predict held-out day before learning its answers',test_from=report['test_from'],updated_on_labels=len(used),predictions=predictions,metrics={})
    for name in totals:
        result['metrics'][name]={scope:{k:dict(n=int(counts[scope][i]),mae=float(totals[name][scope][i]/counts[scope][i]) if counts[scope][i] else None) for i,k in enumerate(SCALES)} for scope in counts}
    atomic_json(Path(directory)/'personalization.json',result)
    return result
