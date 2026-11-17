"""Compatibility smoke tests without private logs or a running server."""
import json
from types import SimpleNamespace
import pytest
from research.data import dt
from research.service import Engine
from research.discovery_cli import main
from research.discovery_service import DiscoveryRuntime


def test_untrained_context_insights_and_preview_do_not_run_legacy_map(tmp_path,monkeypatch):
    from research.insights import InsightEngine
    def forbidden(*args,**kwargs):
        raise AssertionError('Legacy insight pipeline must not run in discovery mode')
    monkeypatch.setattr(InsightEngine,'snapshot',forbidden)
    store=SimpleNamespace(rows=lambda p:[])
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
