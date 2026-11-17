"""Synthetic-only regression tests for the opt-in discovery architecture."""
import copy
from datetime import timedelta
import io
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from research.data import F, BASE, TEXT_DIM, dt
from research.model import HierarchicalLSTM
from research.discovery_model import DiscoveryModel, masked_input, masked_mse, identity_loss, last_observation
from research.discovery_map import DiscoveryMap
from research.discovery_learning import examples_for, source_rows, train_discovery


def inputs():
    torch.manual_seed(5)
    x = torch.randn(2, 2, 2, 12, F)
    x[..., len(BASE):2*len(BASE)] = 1
    return dict(x=x, lengths=torch.full((2,2,2),12), units=torch.full((2,2),2),
                days=torch.full((2,),2), history=torch.randn(2,3,F),
                personal=torch.randn(2,40), activity=torch.tensor([1,23]))


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    return DiscoveryModel(hidden=8)


def test_one_hierarchy_no_fixed_class_head(model):
    assert sum(isinstance(m,torch.nn.LSTM) for m in model.modules())==3
    assert not any('activity' in k or 'personal' in k for k in model.state_dict())
    assert 'activity.weight' in HierarchicalLSTM(hidden=8).state_dict()
    out=model(inputs())
    assert 'activity' not in out
    assert tuple(out['behavior'].shape)==(2,32)
    for s in ('current','daily'):
        assert torch.all((out[s]>=1)&(out[s]<=5))


def test_activity_and_personal_history_not_inputs(model):
    b=inputs(); before=model(b)
    after=model(dict(b,activity=torch.tensor([900,900]),personal=torch.full((2,40),999.)))
    for k in before:
        torch.testing.assert_close(before[k],after[k],atol=0,rtol=0)


def test_scale_predictions_not_behavior_inputs(model):
    b=inputs(); before=model(b)['behavior'].detach().clone()
    with torch.no_grad():
        model.encoder.current.weight.fill_(100)
        model.encoder.daily.weight.fill_(-100)
    torch.testing.assert_close(before,model(b)['behavior'],atol=0,rtol=0)


def test_gradients_reach_all_hierarchies_and_axes(model):
    out=model(inputs())
    sum(out[k].square().mean() for k in ('next','reconstruction','current','daily')).backward()
    for name in ('micro','meso','macro'):
        assert getattr(model.encoder,name).weight_ih_l0.grad.abs().sum().item()>0
    for head in (model.place,model.behavior):
        assert head.weight.grad.abs().sum().item()>0


def test_mask_clears_values_and_presence_without_mutating_original():
    b=inputs(); original=b['x'].clone()
    masked,target,selected=masked_input(b,probability=1)
    idx=last_observation(b); n=len(BASE)
    torch.testing.assert_close(original,b['x'])
    for i in range(n):
        if selected[0,i]:
            assert masked['x'][idx][0,i]==0
            assert masked['x'][idx][0,n+i]==0
    assert torch.all(masked['x'][idx][:,-TEXT_DIM:]==0)
    assert not selected[:,n:2*n].any()
    assert torch.equal(target,original[idx])


def test_missing_targets_and_identities_do_not_create_labels():
    p=torch.tensor([[2.,3.]],requires_grad=True)
    loss=masked_mse(p,torch.tensor([[float('nan'),1.]]),torch.tensor([[0.,1.]]))
    assert float(loss.detach())==4
    loss.backward()
    assert float(p.grad[0,0])==0
    assert float(identity_loss(torch.eye(2),[None,None]))==0


def test_checkpoint_round_trip(model):
    b=io.BytesIO()
    torch.save(dict(weights=model.state_dict(),settings=model.settings),b); b.seek(0)
    state=torch.load(b,weights_only=True)
    other=DiscoveryModel(**state['settings']); other.load_state_dict(state['weights'])
    x=inputs()
    torch.testing.assert_close(model(x)['behavior'],other(x)['behavior'])


def test_unknown_creation_and_read_only_prediction():
    m=DiscoveryMap('behavior',dimension=2)
    assert m.predict([1,0])['candidate_id'] is None
    a=m.observe([1,0],'2026-01-01T10:00:00+09:00')
    before=copy.deepcopy(m.to_dict())
    assert m.predict([1,0])['candidate_id']==a
    assert m.predict([-1,0])['candidate_id'] is None
    assert before==m.to_dict()
    b=m.observe([-1,0],'2026-01-01T11:00:00+09:00')
    assert a!=b and len(m.nodes)==2 and m.edges[(a,b)]==1


def test_no_edges_across_gaps():
    m=DiscoveryMap('behavior',dimension=2)
    m.observe([1,0],'2026-01-01T10:00:00+09:00')
    m.observe([-1,0],'2026-01-01T12:00:01+09:00')
    assert m.edges=={}


def test_capacity_does_not_force_unknown_into_known_identity():
    m=DiscoveryMap('behavior',dimension=2,max_nodes=1)
    m.observe([1,0],'2026-01-01T10:00:00+09:00')
    assert m.observe([-1,0],'2026-01-01T11:00:00+09:00') is None
    assert m.last is None and m.overflow==1
    assert m.observe([-1,0],'2026-01-01T12:00:00+09:00',identity='human')=='human'


def test_annotation_not_geometry_or_confirmation():
    m=DiscoveryMap('behavior',dimension=2)
    a=m.observe([1,0],'2026-01-01T10:00:00+09:00')
    before=m.predict([1,0]); vector=copy.deepcopy(m.nodes[a]['vector'])
    m.annotate(a,'study');m.annotate(a,'desk_work')
    assert m.nodes[a]['vector']==vector
    assert not m.nodes[a]['confirmed'] and m.nodes[a]['confirmed_days']==[]
    assert m.predict([1,0])==before


def test_round_trip_axis_namespaces_and_aliases():
    b=DiscoveryMap('behavior',dimension=2); p=DiscoveryMap('place',dimension=2)
    key=b.observe([1,0],'2026-01-01T10:00:00+09:00')
    assert key!=p.observe([1,0],'2026-01-01T10:00:00+09:00')
    assert DiscoveryMap.from_dict(b.to_dict()).to_dict()==b.to_dict()
    b.aliases['old']=key
    assert b.observe([1,0],'2026-01-02T10:00:00+09:00',identity='old')==key
    assert 'old' not in b.nodes
    assert b.nodes[key]['confirmed_days']==['2026-01-02']


def test_reject_bad_vectors_and_order():
    m=DiscoveryMap('behavior',dimension=2)
    for v in ([0,0],[float('nan'),1],[1,2,3]):
        with pytest.raises(ValueError):m.predict(v)
    m.observe([1,0],'2026-01-01T10:00:00+09:00')
    with pytest.raises(ValueError):m.observe([1,0],'2026-01-01T09:00:00+09:00')
    with pytest.raises(ValueError):m.observe([1,0],'2026-01-01T10:00:00')


def records():
    result=[]
    start=dt('2026-01-05T09:55:00+09:00')
    for d in range(8):
        for h in range(4):
            t=start+timedelta(days=d,hours=h)
            result.append(dict(kind='session',id=f's-{d}-{h}',participant='test',at=t.isoformat(),
                               payload=dict(started_at=(t-timedelta(minutes=2)).isoformat(),ended_at=t.isoformat(),
                                            received_at=t.isoformat(),events=[dict(type='key',time=t.isoformat(),data={}),
                                                dict(type='commit',time=t.isoformat(),data={'text':str(d+h)})])))
    return result,start+timedelta(days=7,hours=5)


def test_fixed_category_answers_do_not_change_sampling_or_inputs():
    rows,at=records()
    original=copy.deepcopy(rows)
    t=rows[0]['at']
    rows.append(dict(kind='narrative',id='activity',participant='test',at=t,
                     payload=dict(question_shown_at=t,question={'kind':'activityCheck'},answer={'label':'game'})))
    assert source_rows(rows)==original
    t1,e1,s1=examples_for(original,at)
    t2,e2,s2=examples_for(rows,at)
    assert s1==s2 and [e['at'] for e in e1]==[e['at'] for e in e2]
    np.testing.assert_array_equal(t1.sample(e1[0]['at'])['x'],t2.sample(e2[0]['at'])['x'])


def test_future_labels_and_targets_respect_split():
    rows,at=records()
    t=dt(rows[0]['at'])
    rows.append(dict(kind='research_event',id='late',participant='test',at=t.isoformat(),
                     payload=dict(kind='context_label',received_at=(at+timedelta(days=1)).isoformat(),
                                  payload=dict(axis='task',class_id='hidden',start=(t-timedelta(minutes=5)).isoformat(),end=t.isoformat()))))
    _,examples,split=examples_for(rows,at)
    assert all(e['labels']['behavior'] is None for e in examples)
    before=dt(split['train_before'])
    for e in examples:
        if e['split']=='train' and e['at']+timedelta(hours=1)>=before:
            assert not e['mask'].any()


def test_unobserved_scale_heads_stay_hidden_and_old_paths_untouched(tmp_path):
    rows,at=records()
    out=tmp_path/'test'/'real'/'discovery'
    legacy=out.parent/'active.pt'; legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b'legacy sentinel')
    report=train_discovery(rows,out,at,epochs=1,hidden=4,device='cpu',evaluate_test=True)
    assert report['status']=='trained'
    assert report['test'] is not None and report['validation']['future_n']>0
    assert legacy.read_bytes()==b'legacy sentinel'
    state=torch.load(out/'active.pt',weights_only=True)
    assert not any('activity' in k for k in state['weights'])
    from research.discovery_service import DiscoveryRuntime
    engine=SimpleNamespace(config={},store=SimpleNamespace(),sample_participant='sample',
                           directory=lambda p,m:tmp_path/p/m,rows=lambda p,m:rows)
    runtime=DiscoveryRuntime(engine)
    live=dt(rows[-1]['at'])+timedelta(minutes=5)
    # The historical model is intentionally not valid before its training time.
    assert runtime.infer('test',live)['status']=='checkpoint_from_future'
    future_session=copy.deepcopy(rows[-1]); future_session['id']='new'
    future_session['at']=at.isoformat()
    future_session['payload'].update(started_at=(at-timedelta(minutes=1)).isoformat(),ended_at=at.isoformat(),received_at=at.isoformat())
    rows.append(future_session)
    prediction=runtime.infer('test',at)
    assert prediction['status']=='ready' and prediction['activity'] is None
    assert prediction['behavior']==prediction['task']
    assert all(v['value'] is None for v in prediction['current'].values())
    assert prediction['behavior']['id'] is None


def test_explicit_architecture_selection(tmp_path):
    from research.service import Engine
    store=SimpleNamespace(rows=lambda p:[])
    legacy=Engine(store,{'research_models':str(tmp_path/'old')})
    assert legacy.discovery is None
    new=Engine(store,{'research_models':str(tmp_path/'new'),'research_architecture':'discovery'})
    assert new.discovery is not None and new.status('test')['architecture']=='discovery'
    assert new.infer('test')['status']=='untrained'
    with pytest.raises(RuntimeError):new.context.train('test',[],dt('2026-01-01T00:00:00+09:00'))
    with pytest.raises(ValueError):Engine(store,{'research_models':str(tmp_path/'bad'),'research_architecture':'typo'})
