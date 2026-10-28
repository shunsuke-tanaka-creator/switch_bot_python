from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from lifelog_server import Store, create_app
from research.data import Timeline, BASE, F, SCALES, dt, monday
from research.model import HierarchicalLSTM
from research.train import batch, train, ablations
from research.service import Engine, export_page, ingest_page
from research.cli import import_file, export_all, sync


def record(kind,id,at,payload):return dict(kind=kind,id=id,at=at.isoformat(),payload=payload,seq=int(at.timestamp()),participant='p001')

def label(at,scope='current',score=3):
    return record('narrative',at.isoformat()+scope,at+timedelta(seconds=30),dict(event_id=at.isoformat()+scope,participant_id='p001',detected_at=at.isoformat(),question_shown_at=at.isoformat(),answered_at=(at+timedelta(seconds=30)).isoformat(),question=dict(kind='currentState' if scope=='current' else 'nightReflection'),answer=dict(free_text=json.dumps(dict.fromkeys(SCALES,score)))))


def synthetic(days=12):
    start=dt('2026-09-01T00:00:00+09:00');rows=[]
    for day in range(days):
        t=start+timedelta(days=day,hours=10)
        for hour in [0,3,6]:
            a=t+timedelta(hours=hour)
            rows.append(record('session',str(day)+str(hour),a-timedelta(seconds=5),dict(started_at=(a-timedelta(minutes=1)).isoformat(),ended_at=(a-timedelta(seconds=5)).isoformat(),events=[dict(time=(a-timedelta(seconds=20)).isoformat(),type='key',data={}),dict(time=(a-timedelta(seconds=10)).isoformat(),type='commit',data=dict(text='研究の記録'))],typing={})))
            rows.append(label(a,score=2+day%3))
        rows.append(label(t+timedelta(hours=12),'daily',score=2+day%3))
    return rows


def test_no_future_or_answer_content_features():
    t=dt('2026-09-10T11:22:00+09:00');rows=synthetic()
    a=Timeline(rows).sample(t)
    changed=rows+[record('session','future',t+timedelta(hours=1),dict(started_at=t.isoformat(),ended_at=(t+timedelta(hours=1)).isoformat(),events=[dict(time=(t+timedelta(minutes=1)).isoformat(),type='commit',data=dict(text='FUTURE'))]))]
    changed.append(label(t+timedelta(minutes=1),score=5))
    b=Timeline(changed).sample(t)
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    # Delayed label not usable as personal baseline until answered.
    l=label(t-timedelta(minutes=1));l['payload']['answered_at']=(t+timedelta(minutes=1)).isoformat();l['at']=l['payload']['answered_at']
    np.testing.assert_array_equal(Timeline([l]).sample(t)['personal'],Timeline([]).sample(t)['personal'])


def test_missing_nonuse_freshness_and_week():
    t=dt('2026-10-26T10:00:00+09:00');start=t-timedelta(minutes=5)
    missing=Timeline([]).features(start,t)
    observed=Timeline([record('heartbeat','h',t-timedelta(seconds=10),dict(logging_enabled=True))]).features(start,t)
    index=BASE.index('key_count')+len(BASE)
    assert missing[index]==0 and observed[index]==1
    assert Timeline([]).sample(t)['days']==1
    assert monday(dt('2026-10-25T23:59:00+09:00')).day==19


def test_hierarchy_gradients_and_ablation():
    sample=Timeline(synthetic(3)).sample(dt('2026-09-03T16:00:00+09:00'))
    b=batch([sample],'cpu',torch.zeros(F),torch.ones(F));model=HierarchicalLSTM(hidden=8)
    out=model(b);out['current'].sum().backward()
    for layer in (model.micro,model.meso,model.macro):assert layer.weight_ih_l0.grad.abs().sum()>0
    assert out['current'].shape==(1,5) and (out['daily']>=1).all() and (out['daily']<=5).all()


def test_training_checkpoint_and_insufficient_heads(tmp_path):
    result=train(synthetic(),tmp_path,epochs=1,hidden=8,evaluate_test=True)
    assert result['status']=='trained' and (tmp_path/'active.pt').exists()
    assert result['train_before']<result['test_from']
    assert result['test']['current']['mood']['n']>0
    model=torch.load(tmp_path/'active.pt',weights_only=True)
    assert model['features'] and model['report']['support']['daily'][0]<10
    with StoreContext() as store:
        engine=Engine(store,dict(research_models=str(tmp_path/'models')))
        assert engine.infer('p001')['status']=='untrained'
    assert train([],tmp_path/'empty',epochs=1)['status']=='insufficient_labels'


class StoreContext:
    def __enter__(self):self.store=Store(':memory:');return self.store
    def __exit__(self,*args):self.store.close()


def test_all_export_replication_corrections_and_heartbeat(tmp_path):
    store=Store(':memory:');app=create_app(dict(ble_enabled=False,allowed_clients=['testclient'],research_models=str(tmp_path)),store)
    with TestClient(app) as c:
        t=datetime.now(timezone.utc)-timedelta(minutes=1)
        r=c.post('/api/heartbeats',json=dict(event_id='h',participant_id='p001',occurred_at=t.isoformat(),source='keyboard',logging_enabled=True))
        assert r.status_code==200
        correction=dict(event_id='c',start=(t-timedelta(hours=1)).isoformat(),end=t.isoformat(),activity='commute',scores={'fatigue':4})
        assert c.post('/api/research/p001/corrections',json=correction).status_code==200
        assert c.post('/api/research/p001/corrections',json={**correction,'scores':{'fatigue':9}}).status_code==422
        page=c.get('/api/export/p001?limit=1').json();snapshot=page['snapshot']
        store.save('context','new','p001',t,dict())
        page2=c.get(f"/api/export/p001?after={page['next_cursor']}&snapshot={snapshot}").json()
        with StoreContext() as target:
            ingest_page(target,page);ingest_page(target,page);ingest_page(target,page2)
            assert len(target.rows('p001'))==2
        assert c.get('/api/research/p001/overview').json()['prediction']['current'] is None
        assert c.get('/research/p001').status_code==200
        graph=c.get('/api/research/p001/pmtt').json();assert any(n.get('source')=='correction' for n in graph['nodes'])
        assert c.get('/research/bad.name').status_code==422


def test_current_and_night_question_cooldown(tmp_path):
    with StoreContext() as store:
        engine=Engine(store,dict(research_models=str(tmp_path)))
        t=dt('2026-10-29T09:00:00+09:00')
        q=engine.pending('p001',t);assert q['kind']=='currentState'
        assert engine.pending('p001',t+timedelta(minutes=1))['id']==q['id']
        store.save('narrative',q['id'],'p001',t+timedelta(minutes=2),dict(question=dict(kind='currentState'),answer=dict(label='current_state')))
        assert engine.pending('p001',t+timedelta(hours=1)) is None
        assert engine.pending('p001',t+timedelta(hours=3,minutes=3))['id']!=q['id']
        assert engine.pending('p001',t.replace(hour=22))['kind']=='nightReflection'
        assert engine.pending('p001',t.replace(hour=2)) is None


def test_legacy_import_idempotent_no_fabricated_labels(tmp_path):
    path=tmp_path/'input.jsonl';t='2026-10-14T12:00:00+09:00'
    path.write_text(json.dumps(dict(file='test',type='key',time=t,data={'keyID':'a'})))
    with StoreContext() as store:
        assert import_file(store,path,'p001')==1
        assert import_file(store,path,'p001')==0
        assert Timeline(store.rows('p001')).labels==[]
        export_all(store,'p001',tmp_path/'all.json')
        with StoreContext() as target:assert import_file(target,tmp_path/'all.json','p001')==1


def test_full_comparison_and_online_adaptation(tmp_path):
    from research.train import personalization_evaluation
    rows=synthetic()
    reports=ablations(rows,tmp_path/'compare',epochs=1,hidden=8)
    assert set(reports)=={'micro','micro_meso','full','no_adaptation','no_text'}
    assert all(r['status']=='trained' for r in reports.values())
    assert len({(r['train_n'],r['test_n'],r['test_from']) for r in reports.values()})==1
    online=personalization_evaluation(rows,tmp_path/'online',epochs=1,hidden=8)
    assert online['predictions'] and online['updated_on_labels']>0
    assert online['metrics']['frozen']['current']['mood']['n']==online['metrics']['updated']['current']['mood']['n']


def test_snapshot_sync_retry_and_late_session(tmp_path):
    import httpx
    calls=[];at=dt('2026-10-14T12:00:00+09:00')
    with StoreContext() as source,StoreContext() as target:
        source.save('heartbeat','h','p001',at,dict(logging_enabled=True))
        def handler(request):
            page=export_page(source,'p001',int(request.url.params.get('after',0)))
            calls.append(request);return httpx.Response(200,json=page)
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            assert sync(target,'http://mac','p001',tmp_path/'cursor.json',client)['inserted']==1
            assert sync(target,'http://mac','p001',tmp_path/'cursor.json',client)['inserted']==0
        assert len(target.rows('p001'))==1
    row=record('session','delayed',at+timedelta(minutes=1),dict(started_at=at.isoformat(),ended_at=(at+timedelta(minutes=1)).isoformat(),received_at=(at+timedelta(minutes=3)).isoformat(),events=[dict(type='key',time=(at+timedelta(seconds=10)).isoformat(),data={})]))
    v=Timeline([row]).features(at,at+timedelta(seconds=30))
    assert v[BASE.index('key_count')+len(BASE)]==0


def test_warm_inference_gates_each_head(tmp_path):
    from shutil import copyfile
    rows=synthetic();trained=tmp_path/'trained'
    assert train(rows,trained,epochs=1,hidden=8)['status']=='trained'
    with StoreContext() as store:
        for r in rows:store.save(r['kind'],r['id'],'p001',dt(r['at']),r['payload'])
        engine=Engine(store,dict(research_models=str(tmp_path/'models')))
        directory=engine.directory('p001');directory.mkdir(parents=True)
        copyfile(trained/'active.pt',directory/'active.pt')
        at=dt('2026-09-12T16:00:01+09:00')
        result=engine.infer('p001',at)
        assert result['status']=='ready'
        assert 1<=result['current']['mood']['value']<=5
        assert result['daily']['mood']['value'] is None
        assert result['activity'] is None  # no activity labels => no random activity returned
