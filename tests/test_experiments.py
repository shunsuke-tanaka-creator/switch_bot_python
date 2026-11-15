"""Leakage, participant isolation, staged optimization and opt-in regressions.

Only the repository's artificial cohort and in-memory arrays are used.
"""
from datetime import timedelta
import copy
import json
import subprocess
import sys
import numpy as np
import pytest
import torch
from research.data import BASE, F, Label, LazySample, Timeline, dt
from research.train import batch
from research.experiment_config import ExperimentConfig
from research.experiment_data import Example, cohort, chronological_split, eligible_updates, sensor_timelines
from research.experiment_metrics import ReferencePredictors, state_metrics
from research.experiment_sample import sample_cohort
from research.experimental_model import ExperimentalModel, configure_stage
from research.experiments import (run_experiment, load_experiment, model_batch, set_counts,
                                  supervised_loss, step, prequential)
from research.ssl import observation_mask, corrupt_observed, masked_mse, SSLHeads, ssl_anchors


def example(person='a', hour=10, score=3., scope='current', available=None, ident='answer'):
    at = dt(f'2026-09-07T{hour:02d}:00:00+09:00')
    l = Label(at, available or at+timedelta(seconds=1), scope, np.full(5, score, np.float32), id=ident)
    return Example(person, l, LazySample(Timeline([]), at, scope))


def test_macro_equal_participant_weight_and_missing_scales():
    a, b = example('a', score=1), example('b', score=5)
    records = [('a', a.label, np.full(5, 1.))]*9 + [('b', b.label, np.full(5, 1.))]
    result = state_metrics(records)
    assert result['MAE_macro'] == 2.  # Not micro MAE=.4.
    assert result['scales']['mood'] == dict(MAE_macro=2., participants=2, n=10)
    b.label.values[1:] = np.nan
    result = state_metrics(records)
    assert result['MAE_macro'] == 2.
    assert result['scales']['fatigue']['participants'] == 1
    assert result['by_scope']['daily']['MAE_macro'] is None
    json.dumps(result, allow_nan=False)


def test_metrics_reject_nonfinite_predictions():
    e = example()
    with pytest.raises(ValueError, match='finite'):
        state_metrics([(e.participant, e.label, np.full(5, np.nan))])


def test_reference_predictors_use_own_scope_and_received_past_only():
    train = [example('a', 9, 2), example('a', 10, 4), example('b', 10, 5), example('a', 9, 1, 'daily')]
    earlier = example('a', 11, 3)
    delayed = example('a', 11, 5, available=dt('2026-09-07T15:00:00+09:00'))
    future = example('a', 15, 5)
    target = example('a', 12, 1)
    refs = ReferencePredictors(train, train+[earlier, delayed, future, target])
    pred = refs.predict(target)
    np.testing.assert_array_equal(pred['personal_mean'], 3)
    np.testing.assert_array_equal(pred['last_answer'], 3)
    target.label.at = target.label.at.replace(hour=10) + timedelta(minutes=1)
    np.testing.assert_array_equal(refs.predict(target)['time_of_day_mean'], 4)
    # Cold start has the training-only cohort fallback, separately for each scope.
    target.participant = 'new'
    np.testing.assert_allclose(refs.predict(target)['personal_mean'], (2+4+5)/3)
    earlier.label.values[:] = 1
    assert refs.predict(target)['last_answer'][0] == pytest.approx((2+4+5)/3)


def test_ridge_never_fits_evaluation_targets():
    train = [example(hour=8, score=2), example(hour=9, score=4)]
    target = example(hour=11, score=1)
    first = ReferencePredictors(train, train+[target], ridge=True).predict(target)['ridge']
    target.label.values[:] = 5
    second = ReferencePredictors(train, train+[target], ridge=True).predict(target)['ridge']
    np.testing.assert_array_equal(first, second)


def test_cohort_isolation_and_colliding_label_ids():
    timelines, items = cohort(sample_cohort(5))
    assert len(timelines) == 2
    assert len({e.key for e in items}) == len(items)
    for e in items:
        assert all(r['participant'] == e.participant for r in e.sample.timeline.rows)
    at = dt('2026-09-09T23:00:00+09:00')
    original = timelines['sample-a'].sample(at)
    changed = copy.deepcopy(sample_cohort(5))
    for row in changed:
        if row['participant'] == 'sample-b' and row['kind'] == 'narrative':
            row['payload']['answer']['free_text'] = json.dumps(dict(mood=5))
    after = cohort(changed)[0]['sample-a'].sample(at)
    for key in original:
        np.testing.assert_array_equal(original[key], after[key])


def test_split_excludes_boundary_and_delayed_training_answers():
    _, examples = cohort(sample_cohort())
    cutoff = dt('2026-09-13T00:00:00+09:00')
    delayed = copy.deepcopy(examples[0]); delayed.label.id = 'late'; delayed.label.available = cutoff
    train, valid, test, bounds = chronological_split(examples+[delayed], cutoff, '2026-09-15T00:00:00+09:00')
    assert delayed not in train
    assert all(e.label.at < cutoff and e.label.available < cutoff for e in train)
    assert all(cutoff <= e.label.at < dt(bounds['test_from']) for e in valid)
    assert all(e.label.at >= dt(bounds['test_from']) for e in test)
    assert delayed in eligible_updates(examples+[delayed], {e.key for e in train}, cutoff+timedelta(seconds=1))
    assert delayed not in eligible_updates(examples+[delayed], {e.key for e in train}, cutoff)


@pytest.mark.parametrize('kwargs', [dict(train_before='2026-09-13'),
    dict(train_before='2026-09-13T00:00:00', test_from='2026-09-15T00:00:00+09:00'),
    dict(pretrain_before='2026-10-01T00:00:00+09:00')])
def test_split_rejects_invalid_or_leaking_cutoffs(kwargs):
    with pytest.raises(ValueError):
        chronological_split(cohort(sample_cohort())[1], **kwargs)


def test_ssl_masks_only_observed_nonpadding_nontime_features():
    values = torch.zeros(1, 2, 12, F)
    n = len(BASE)
    values[..., :n] = 8
    values[..., n:2*n] = 1
    values[0, 0, 0, n] = 0
    lengths = torch.tensor([[2, 0]])
    observed = observation_mask(values, lengths)
    hidden, selected = corrupt_observed(values, observed, 1.)
    assert not selected[0, 0, 0, 0]
    assert not selected[..., n:2*n].any()
    assert not selected[0, 1].any() and not selected[0, 0, 2:].any()
    assert not selected[..., BASE.index('hour_sin')].any()
    assert hidden[0, 0, 0, 1] == 0 and hidden[0, 0, 0, n+1] == 0
    # Originally missing NaNs never enter the loss.
    target = values.clone(); target[~observed] = float('nan')
    prediction = torch.zeros_like(target, requires_grad=True)
    loss = masked_mse(prediction, target, observed); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(prediction.grad).all()
    assert prediction.grad[~observed].abs().sum() == 0


def test_ssl_gradients_reach_all_three_hierarchies():
    timelines, _ = cohort(sample_cohort(1))
    timeline = sensor_timelines(timelines)['sample-a']
    at = dt('2026-09-07T10:00:00+09:00')
    model = ExperimentalModel(hidden=4)
    b = batch([timeline.sample(at)], 'cpu', torch.zeros(F), torch.ones(F))
    future = torch.ones(1, 1, F)
    observed = observation_mask(future)
    loss, _ = SSLHeads(4, (5,)).loss(model, b, future, observed, 1.)
    loss.backward()
    for layer in (model.micro, model.meso, model.macro):
        assert layer.weight_ih_l0.grad.abs().sum() > 0
    assert model.current.weight.grad is None


def test_ssl_no_answer_boundaries_or_future_cutoff_targets():
    timelines, _ = cohort(sample_cohort())
    timeline = sensor_timelines(timelines)['sample-a']
    assert not timeline.labels and not timeline.corrections
    cutoff = dt('2026-09-10T10:30:00+09:00')
    anchors = ssl_anchors(timeline, cutoff, (5, 30), max_samples=6)
    assert anchors and all(t+timedelta(minutes=30) < cutoff for t in anchors)
    assert len(anchors) <= 6
    with pytest.raises(ValueError):
        ssl_anchors(timeline, cutoff, (31,))


def test_head_stage_freezes_encoder_then_upper_stage_keeps_lower_layers_frozen():
    _, examples = cohort(sample_cohort(1))
    model = ExperimentalModel(hidden=4, participants=['sample-a', 'sample-b'])
    before = copy.deepcopy(model.state_dict())
    params = configure_stage(model, 'heads')
    mean, std = torch.zeros(F), torch.ones(F)
    step(model, torch.optim.SGD(params, lr=.05), examples[:1], mean, std)
    for key, value in before.items():
        if not key.startswith(('current.', 'daily.', 'activity.')):
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    assert not torch.equal(model.current.weight, before['current.weight'])
    configure_stage(model, 'upper')
    assert not model.micro.weight_ih_l0.requires_grad and not model.meso.weight_ih_l0.requires_grad
    assert model.macro.weight_ih_l0.requires_grad and model.top_work[0].weight.requires_grad
    assert not model.personal[0].weight.requires_grad


def test_personal_bias_sparse_support_shrinks_more_and_does_not_change_encoder():
    model = ExperimentalModel(hidden=4, participants=['sparse', 'dense'], partial_pooling=True, bias_l2=10.)
    with torch.no_grad():
        model.current.weight.zero_(); model.current.bias.zero_()
    sparse, dense = example('sparse', score=4.), example('dense', score=4.)
    items = [sparse] + [dense]*20
    set_counts(model, items)
    params = configure_stage(model, 'bias')
    b = model_batch(model, [sparse, dense], torch.zeros(F), torch.ones(F))
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.LBFGS(params, max_iter=30, tolerance_grad=1e-9)
    def closure():
        optimizer.zero_grad()
        loss = supervised_loss(model, [sparse, dense], b)
        loss.backward()
        return loss
    optimizer.step(closure)
    np.testing.assert_allclose(model.personal_bias.detach().numpy()[0], 1/11, atol=1e-4)
    np.testing.assert_allclose(model.personal_bias.detach().numpy()[1], 20/30, atol=1e-4)
    for key in before:
        if key != 'personal_bias':
            torch.testing.assert_close(model.state_dict()[key], before[key], rtol=0, atol=0)
    # Unseen participant has exactly the shared prediction, not another person's bias.
    unseen = model_batch(model, [example('unseen')], torch.zeros(F), torch.ones(F))
    torch.testing.assert_close(model(unseen)['current'], torch.full((1, 5), 3.))


def test_bias_update_does_not_modify_other_participants():
    model = ExperimentalModel(hidden=4, participants=['a', 'b'], partial_pooling=True)
    one = example('a', score=5)
    configure_stage(model, 'bias')
    step(model, torch.optim.SGD([model.personal_bias], lr=.1), [one], torch.zeros(F), torch.ones(F))
    assert model.personal_bias[0].abs().sum() > 0
    assert model.personal_bias[1].abs().sum() == 0


def test_prequential_predicts_before_answer_and_handles_simultaneous_participants():
    model = ExperimentalModel(hidden=4, participants=['a', 'b'], partial_pooling=True)
    past = example('a', 8, 2, ident='past')
    first = example('a', 10, 1, ident='same')
    simultaneous = example('b', 10, 5, ident='same')
    later = example('a', 12, 3, ident='later')
    all_examples = [past, first, simultaneous, later]
    set_counts(model, [past])
    args = (torch.zeros(F), torch.ones(F), .01)
    one = prequential(model, all_examples, [past], all_examples[1:], *args)
    first.label.values[:] = 5
    two = prequential(model, all_examples, [past], all_examples[1:], *args)
    for index in (0, 1):
        for mode in ('frozen', 'personal_bias', 'full_finetuning'):
            assert one['predictions'][index][mode] == two['predictions'][index][mode]
        assert one['predictions'][index]['updated_on_labels'] == 0
    assert one['predictions'][2]['updated_on_labels'] == 2
    assert one['predictions'][2]['personal_bias'] != two['predictions'][2]['personal_bias']
    assert one['predictions'][2]['frozen'] == two['predictions'][2]['frozen']
    for pred in one['predictions']:
        assert dt(pred['latest_update_available']) < dt(pred['at'])


def test_disabled_defaults_have_no_filesystem_side_effects(tmp_path):
    path = tmp_path/'disabled'
    with pytest.raises(ValueError, match='disabled'):
        run_experiment([], path)
    assert not path.exists()


@pytest.mark.parametrize('config', [ExperimentConfig(enabled_prequential=True),
    ExperimentConfig(enabled_evaluation=True, upper_finetune=True),
    ExperimentConfig(enabled_evaluation=True, finetune_learning_rate=.1),
    ExperimentConfig(enabled_evaluation=True, bias_l2=0)])
def test_invalid_opt_in_combinations_rejected(config):
    with pytest.raises(ValueError):
        config.validate()


def test_full_opt_in_pipeline_roundtrip_and_production_isolation(tmp_path):
    (tmp_path/'active.pt').write_bytes(b'production-sentinel')
    config = ExperimentConfig(enabled_evaluation=True, enabled_ridge=True, enabled_ssl=True,
        enabled_partial_pooling=True, enabled_prequential=True, upper_finetune=True,
        epochs=1, ssl_epochs=1, finetune_epochs=1, hidden=4, ssl_max_samples=2,
        pretrain_before='2026-09-11T00:00:00+09:00')
    report = run_experiment(sample_cohort(), tmp_path, config)
    assert report['status'] == 'trained' and report['production_promoted'] is False
    assert (tmp_path/'active.pt').read_bytes() == b'production-sentinel'
    assert not (tmp_path/'status.json').exists()
    assert [s['stage'] for s in report['stages']] == ['heads', 'upper']
    assert set(report['baselines']['test']) == {'personal_mean', 'last_answer', 'time_of_day_mean', 'ridge'}
    assert set(report['prequential']['metrics']) == {'frozen', 'personal_bias', 'full_finetuning'}
    assert report['test']['participants'] == 2
    assert dt(report['ssl']['latest_target_end']) < dt(report['boundaries']['pretrain_before'])
    json.dumps(report, allow_nan=False)
    model, mean, std, restored = load_experiment(tmp_path/'experiment.pt')
    assert restored['test'] == report['test']
    from research.experiments import evaluate
    _, examples = cohort(sample_cohort())
    test = chronological_split(examples, pretrain_before=config.pretrain_before)[2]
    assert evaluate(model, test, mean, std) == report['test']
    saved = torch.load(tmp_path/'experiment.pt', weights_only=True)
    assert 'version' not in saved  # Serving compatibility check rejects this artifact.


def test_changing_test_labels_cannot_change_training_or_normalization(tmp_path):
    config = ExperimentConfig(enabled_evaluation=True, epochs=1, hidden=4)
    rows = [r for r in sample_cohort() if r['participant'] == 'sample-a']
    first = run_experiment(rows, tmp_path/'one', config)
    modified = copy.deepcopy(rows)
    for row in modified:
        if row['kind'] == 'narrative' and dt(row['payload']['question_shown_at']) >= dt(first['boundaries']['test_from']):
            row['payload']['answer']['free_text'] = json.dumps(dict.fromkeys(('mood','fatigue','stress','concentration','sleepiness'), 5))
    run_experiment(modified, tmp_path/'two', config)
    a, b = [torch.load(tmp_path/name/'experiment.pt', weights_only=True) for name in ('one', 'two')]
    assert a['mean'] == b['mean'] and a['std'] == b['std']
    for key in a['weights']:
        torch.testing.assert_close(a['weights'][key], b['weights'][key], rtol=0, atol=0)


def test_cli_help_does_not_import_optional_torch():
    code = """
import sys
class BlockTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch': raise RuntimeError('optional ML imported')
sys.meta_path.insert(0, BlockTorch())
from research.cli import main
sys.argv = ['research.cli', '--help']
main()
"""
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'experiment' in result.stdout


def test_ssl_ignores_delayed_reception_and_post_cutoff_sensor_rows():
    cutoff = dt('2026-09-09T00:00:00+09:00')
    base = [r for r in sample_cohort(3) if r['participant'] == 'sample-a']
    modified = copy.deepcopy(base)
    for row in modified:
        if row['kind'] == 'heartbeat' and dt(row['at']) >= cutoff:
            row['payload']['logging_enabled'] = False
    modified.append(dict(participant='sample-a', id='late-session', kind='session',
                         at='2026-09-07T10:00:00+09:00', payload=dict(
                             started_at='2026-09-07T09:59:00+09:00', ended_at='2026-09-07T10:00:00+09:00',
                             received_at='2026-09-10T10:00:00+09:00',
                             events=[dict(time='2026-09-07T09:59:30+09:00', type='key', data={})])))
    timelines = [sensor_timelines(cohort(rows)[0])['sample-a'] for rows in (base, modified)]
    anchors = [ssl_anchors(t, cutoff, (5, 30), max_samples=2) for t in timelines]
    assert anchors[0] == anchors[1]
    for at in anchors[0]:
        left, right = [t.sample(at) for t in timelines]
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
        for h in (5, 30):
            target = [t.features(at+timedelta(minutes=h-5), at+timedelta(minutes=h)) for t in timelines]
            np.testing.assert_array_equal(*target)


def test_cli_fixture_and_readonly_snapshot_paths(tmp_path, monkeypatch):
    import argparse
    from research.experiment_cli import add_parser, run
    from lifelog_server import Store
    import research.experiments as runner
    seen = []
    def fake_run(rows, directory, config):
        seen.append(rows)
        return dict(status='trained', test=dict(MAE_macro=0.))
    monkeypatch.setattr(runner, 'run_experiment', fake_run)
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers(dest='command'))
    args = parser.parse_args(['experiment', '--sample-fixture', '--enable-partial-pooling'])
    args.db = str(tmp_path/'does-not-exist.db')
    assert run(args)['status'] == 'trained'
    assert not (tmp_path/'does-not-exist.db').exists()
    db = tmp_path/'sample.db'
    with_store = Store(db)
    try:
        for r in sample_cohort(1):
            with_store.save(r['kind'], r['id'], r['participant'], dt(r['at']), r['payload'])
    finally:
        with_store.close()
    before = db.read_bytes()
    args = parser.parse_args(['experiment', '--participants', 'sample-a', '--enable-evaluation'])
    args.db = str(db)
    assert run(args)['status'] == 'trained'
    assert db.read_bytes() == before
    assert {r['participant'] for r in seen[-1]} == {'sample-a'}
