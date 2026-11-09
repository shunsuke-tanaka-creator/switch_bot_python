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
from research.data_modes import rows_for_mode, source_identity
from research.sample_data import build_records, generate_sample, load_calendar


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
        q=engine.pending('p001',t);assert q['kind']=='weatherMood'
        assert engine.pending('p001',t+timedelta(minutes=1))['id']==q['id']
        store.save('narrative',q['id'],'p001',t+timedelta(minutes=2),dict(question=dict(kind='currentState'),answer=dict(label='current_state')))
        assert engine.pending('p001',t+timedelta(minutes=30)) is None
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


def test_data_modes_keep_real_separate_and_combine_chronologically():
    with StoreContext() as store:
        first=dt('2026-10-01T09:00:00+09:00');second=first+timedelta(hours=1)
        store.save('heartbeat','real','p001',second,dict(logging_enabled=True))
        store.save('heartbeat','sample','p001-sample',first,dict(logging_enabled=True))
        real=rows_for_mode(store,'p001','real')
        sample=rows_for_mode(store,'p001','sample')
        combined=rows_for_mode(store,'p001','combined')
        assert [r['participant'] for r in real]==['p001']
        assert [r['participant'] for r in sample]==['p001-sample']
        assert [r['id'] for r in combined]==['real','sample']


def test_sample_generation_is_deterministic_idempotent_and_manifest_guarded(tmp_path):
    calendar_path=Path('data/private/p001-calendar-tags.json')
    calendar,_=load_calendar(calendar_path)
    now=dt('2026-11-01T17:00:00+09:00')
    assert build_records(calendar,include_today=True,now=now)==build_records(calendar,include_today=True,now=now)
    manifest=tmp_path/'sample-manifest.json'
    with StoreContext() as store:
        one=generate_sample(calendar_path,store=store,manifest_path=manifest,include_today=True,now=now)
        two=generate_sample(calendar_path,store=store,manifest_path=manifest,include_today=True,now=now+timedelta(hours=1))
        assert one['inserted']==1053 and two['inserted']==0 and two['existing']==1053
        assert len(store.rows('p001'))==0 and len(store.rows('p001-sample'))==1053
        labels=Timeline(store.rows('p001-sample')).labels
        values={tuple(l.values) for l in labels if l.scope in ('current','daily')}
        assert len(values)>10 and all(np.isfinite(l.values).all() for l in labels if l.scope in ('current','daily'))
        altered=tmp_path/'altered.json'
        value=json.loads(calendar_path.read_text());value['days'][0]['tags']=['none']
        altered.write_text(json.dumps(value))
        with pytest.raises(RuntimeError,match='manifest differs'):
            generate_sample(altered,store=store,manifest_path=manifest,include_today=True,now=now)


def test_model_paths_are_mode_separated_with_real_legacy_fallback(tmp_path):
    with StoreContext() as store:
        engine=Engine(store,dict(research_models=str(tmp_path)))
        assert engine.directory('p001','real')==tmp_path/'p001'/'real'
        assert engine.directory('p001','sample')==tmp_path/'p001'/'sample'
        legacy=tmp_path/'p001';legacy.mkdir()
        (legacy/'status.json').write_text(json.dumps(dict(status='trained',fingerprint='legacy')))
        assert engine.status('p001','real')['fingerprint']=='legacy'
        engine.directory('p001','real').mkdir()
        (engine.directory('p001','real')/'status.json').write_text(json.dumps(dict(status='trained',fingerprint='new')))
        assert engine.status('p001','real')['fingerprint']=='new'


def test_training_fingerprint_tracks_mode_source_seq_and_manifest(tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text('{"version":1}')
    with StoreContext() as store:
        at=dt('2026-10-01T09:00:00+09:00')
        store.save('session','real','p001',at,dict(started_at=at.isoformat(),ended_at=at.isoformat(),events=[]))
        store.save('session','sample','p001-sample',at,dict(started_at=at.isoformat(),ended_at=at.isoformat(),events=[]))
        real=source_identity(store,'p001','real','p001-sample',manifest)
        combined=source_identity(store,'p001','combined','p001-sample',manifest)
        assert real['fingerprint']!=combined['fingerprint']
        assert set(combined['source_max_seq'])=={'p001','p001-sample'}
        before=combined['fingerprint']
        manifest.write_text('{"version":2}')
        assert source_identity(store,'p001','combined','p001-sample',manifest)['fingerprint']!=before


def test_http_data_mode_defaults_validation_and_explicit_training(tmp_path):
    store=Store(':memory:')
    at=datetime.now(timezone.utc)-timedelta(minutes=2)
    store.save('narrative','sample-answer','p001-sample',at,dict(
        question=dict(kind='currentState'),question_shown_at=at.isoformat(),
        answer=dict(free_text=json.dumps(dict.fromkeys(SCALES,3)))))
    app=create_app(dict(ble_enabled=False,allowed_clients=['testclient'],research_models=str(tmp_path)),store)
    with TestClient(app) as c:
        assert c.get('/api/research/p001/overview').json()['data_mode']=='real'
        sample=c.get('/api/research/p001/overview?data_mode=sample').json()
        assert sample['contains_sample'] and sample['answers'][0]['kind']=='currentState'
        assert c.get('/api/research/p001/insights?data_mode=bad').status_code==422
        html=c.get('/research/p001').text
        assert 'id="dataMode"' in html and 'サンプル込みの読み取り専用表示' in html
        app.state.research.train=lambda participant,data_mode_:dict(status='called',participant=participant,data_mode=data_mode_)
        assert c.post('/api/research/p001/train',json={'data_mode':'combined'}).json()=={
            'status':'called','participant':'p001','data_mode':'combined'}
