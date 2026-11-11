"""Model-path regressions. Requires the ML/dev environment; never uses user data."""
import numpy as np
import torch
from research.data import Timeline, BASE, F, FEATURES, LazySample, dt, materialize
from research.train import batch, normalization
from research.service import Engine
from lifelog_server import Store


def test_streaming_normalization_matches_observed_training_reference():
    samples=[]
    rng=np.random.default_rng(17)
    for _ in range(3):
        x=np.zeros((7,48,12,F),np.float32)
        x[0,0,:3,:len(BASE)]=rng.normal(size=(3,len(BASE)))
        x[0,0,:3,len(BASE):2*len(BASE)]=rng.integers(0,2,size=(3,len(BASE)))
        lengths=np.zeros((7,48),np.int64);lengths[0,0]=3
        samples.append((None,dict(x=x,lengths=lengths)))
    mean,std=normalization(samples)
    values=np.concatenate([s['x'][np.arange(12)[None,None,:]<s['lengths'][...,None]] for _,s in samples])
    for j in range(len(BASE)):
        observed=values[:,j][values[:,j+len(BASE)]>0]
        np.testing.assert_allclose(mean[j],observed.mean() if len(observed) else 0,atol=1e-6)
        np.testing.assert_allclose(std[j],max(observed.std(),.1) if len(observed) else 1,atol=1e-6)
    np.testing.assert_array_equal(mean[len(BASE):],0)
    np.testing.assert_array_equal(std[len(BASE):],1)


def test_lazy_and_materialized_batches_are_identical():
    timeline=Timeline([])
    lazy=LazySample(timeline,dt('2026-11-02T01:00:00+09:00'))
    args=('cpu',torch.zeros(F),torch.ones(F))
    expected=batch([materialize(lazy)],*args)
    actual=batch([lazy],*args)
    for key in expected:torch.testing.assert_close(actual[key],expected[key])


def test_old_feature_checkpoint_is_rejected_without_overwrite(tmp_path):
    store=Store(':memory:')
    try:
        engine=Engine(store,dict(research_models=str(tmp_path)))
        path=engine.directory('p001')/'active.pt';path.parent.mkdir(parents=True)
        torch.save(dict(version='hierarchical-v1',features=FEATURES),path)
        original=path.read_bytes()
        assert engine.infer('p001')['status']=='incompatible_checkpoint'
        assert path.read_bytes()==original
    finally:store.close()
