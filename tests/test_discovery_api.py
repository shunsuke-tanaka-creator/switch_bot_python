"""Compatibility smoke tests without private logs or a running server."""
import json
from datetime import timedelta  # 追加: 合成sourceの同時刻履歴と観測鮮度を検証する。
from threading import RLock
from types import SimpleNamespace
import pytest
from research.data import dt
from research.service import Engine
from research.discovery_cli import main
from research.discovery_service import DiscoveryRuntime, DiscoveryInsights  # 追加: combined比較の回帰。


def test_untrained_context_insights_and_preview_do_not_run_legacy_map(tmp_path,monkeypatch):
    from research.insights import InsightEngine
    def forbidden(*args,**kwargs):
        raise AssertionError('Legacy insight pipeline must not run in discovery mode')
    monkeypatch.setattr(InsightEngine,'snapshot',forbidden)
    store=SimpleNamespace(rows=lambda p:[],lock=RLock())
    engine=Engine(store,{'research_models':str(tmp_path),'research_architecture':'discovery'})
    at=dt('2026-01-05T12:00:00+09:00')
    context=engine.context.snapshot('test',at)
    assert context['architecture']=='discovery' and context['discovery_maps']=={}
    insight=engine.insights.snapshot('test',at)
    assert insight['architecture']=='discovery'
    assert insight['patterns']['nodes']==[]
    assert insight['exercise']['status']=='disabled'
    assert insight['fatigue']['acute'] is False
    preview=engine.proposal_preview('test',at=at)
    assert preview['preview'] is True
    assert preview['exercise']['status']=='disabled'


# 追加: 同時刻回答・同端末入力を持つ本人/sampleを、Legacyと同じ観測鮮度で分離する。
@pytest.mark.parametrize('real_age,selected',[(30,'test'),(90,'test-sample')])
def test_combined_insights_compare_each_source_without_mixing(real_age,selected):
    from research.baseline import conditional, weekday_trends
    from research.topology import typing_windows
    from research.insights import typing_change
    at=dt('2026-11-24T15:00:00+09:00')
    groups={}
    for owner in ('test','test-sample'):
        records=[]
        def record(kind,identifier,t,payload):
            return dict(kind=kind,id=identifier,at=t.isoformat(),participant=owner,payload=payload)
        for day in range(1,50):
            cutoff=(at-timedelta(days=day)).replace(hour=12)
            if cutoff.weekday() not in (0,1,2):
                continue
            value=(1 if cutoff.weekday()==0 else 2) if owner=='test' else (5 if cutoff.weekday()==0 else 3)
            records.append(record('narrative',f'answer-{day}',cutoff+timedelta(seconds=1),
                                  dict(question=dict(kind='currentState'),question_shown_at=cutoff.isoformat(),
                                       answer=dict(free_text=json.dumps(dict(fatigue=value))))))
        # 追加: 同時刻回答の上書きに加え、異なるcutoff回答の混合も検証する。
        cutoff=(at-timedelta(days=1)).replace(hour=12,minute=2 if owner=='test' else 4)
        records.append(record('narrative',owner+'-extra',cutoff+timedelta(seconds=1),
                              dict(question=dict(kind='currentState'),question_shown_at=cutoff.isoformat(),
                                   answer=dict(free_text=json.dumps(dict(fatigue=1 if owner=='test' else 5))))))
        for day in range(5):
            start=(at-timedelta(days=day)).replace(hour=12,minute=0)
            events=[]
            for i in range(50):
                t=start+timedelta(seconds=i*3)
                events.append(dict(type='kana',time=t.isoformat(),data=dict(input_mode='kana')))
                if i<(2 if owner=='test' else 20):
                    events.append(dict(type='delete',time=(t+timedelta(seconds=1)).isoformat(),data={}))
            records.append(record('session',f'{owner}-session-{day}',start+timedelta(minutes=3),
                                  dict(session_id=f'{owner}-session-{day}',device_id='shared-phone',events=events)))
        age=real_age if owner=='test' else 10
        records.append(record('heartbeat',owner+'-fresh',at-timedelta(minutes=age),{}))
        groups[owner]=records
    rows=groups['test']+groups['test-sample']
    runtime=SimpleNamespace(config={},engine=SimpleNamespace(rows=lambda p,mode:rows),load=lambda p,mode:None)
    engine=DiscoveryInsights(runtime)
    result=engine.snapshot('test',at,rows=rows,data_mode='combined')
    assert result['sensor_source']==selected
    expected=dict(baseline=conditional(groups[selected],at),trends=weekday_trends(groups[selected],at),
                  typing=typing_change(typing_windows(groups[selected],at),at))
    assert expected['baseline']['fatigue']['status']=='available'
    assert expected['trends']
    assert expected['typing']['status']=='available'
    for name,value in expected.items():
        assert result[name]==value
    assert result['typing']['inserts']==50
    assert result['typing']['correction_rate']==(0.04 if selected=='test' else 0.4)
    assert result['baseline']!=conditional(rows,at)
    assert result['trends']!=weekday_trends(rows,at)
    assert result['typing']!=typing_change(typing_windows(rows,at),at)
    assert engine.snapshot('test',at,rows=rows,data_mode='combined')==result
    assert result['support']['reason']=='combined_history_not_supported'
    assert result['support']['delivery_mode']=='shadow' and not result['support']['eligible']
    assert result['exercise']['status']=='disabled'


# 追加: 回答を追加せずsourceだけ増えた場合も曜日傾向のcacheを更新する。
def test_combined_insights_cache_tracks_heartbeat_only_sources():
    at=dt('2026-11-24T15:00:00+09:00')
    runtime=SimpleNamespace(config={},load=lambda p,mode:None)
    engine=DiscoveryInsights(runtime)
    assert engine.snapshot('test',at,rows=[],data_mode='combined')['trends']==[]
    rows=[dict(kind='heartbeat',id='sample',at=(at-timedelta(minutes=1)).isoformat(),
               participant='test-sample',payload={})]
    sample=engine.snapshot('test',at,rows=rows,data_mode='combined')
    assert sample['sensor_source']=='test-sample'
    rows.append(dict(kind='heartbeat',id='real',at=(at-timedelta(minutes=30)).isoformat(),
                     participant='test',payload={}))
    real=engine.snapshot('test',at,rows=rows,data_mode='combined')
    assert real['sensor_source']=='test'
    assert real['trends']==[] and real['typing']['status']=='insufficient_data'


def test_cache_return_survives_concurrent_eviction(tmp_path):
    engine=SimpleNamespace(config={},store=None,directory=lambda p,m:tmp_path)
    runtime=DiscoveryRuntime(engine)
    path=tmp_path/'discovery'/'active.pt'
    path.parent.mkdir();path.write_bytes(b'not loaded because the snapshot is cached')
    stat=path.stat();key=(str(path),stat.st_mtime_ns,stat.st_size)
    snapshot=object()
    class EvictingCache(dict):
        def __contains__(self,k):
            found=super().__contains__(k)
            self.clear()
            return found
        def get(self,k,default=None):
            value=super().get(k,default)
            self.clear()
            return value
    runtime.cache=EvictingCache({key:snapshot})
    assert runtime.load('test') is snapshot
    assert runtime.cache=={}


@pytest.mark.parametrize('problem',['incomplete','mixed_participants','unidentified_checkpoint','legacy_status'])
def test_cli_rejects_unsafe_input_without_overwriting(tmp_path,problem):
    source=tmp_path/'export.json'
    output=tmp_path/'models';output.mkdir()
    payload=dict(format='lifelog-all-v1',participant='test',records=[],next_cursor=None)
    if problem=='incomplete':payload['next_cursor']=10
    if problem=='mixed_participants':payload['records']=[dict(participant='other')]
    if problem in ('unidentified_checkpoint','legacy_status'):
        (output/'active.pt').write_bytes(b'legacy checkpoint sentinel')
    if problem=='legacy_status':
        (output/'status.json').write_text(json.dumps(dict(status='trained')),encoding='utf-8')
    source.write_text(json.dumps(payload),encoding='utf-8')
    with pytest.raises(SystemExit) as error:
        main(['--export',str(source),'--output',str(output)])
    assert error.value.code==2
    if (output/'active.pt').exists():
        assert (output/'active.pt').read_bytes()==b'legacy checkpoint sentinel'
    assert not (output/'candidate.pt').exists()
