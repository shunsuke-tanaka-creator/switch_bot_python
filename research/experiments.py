"""Offline, explicitly enabled experiments. Never promotes a production checkpoint."""
from dataclasses import asdict
from pathlib import Path
import copy
import fcntl
import os
import random
import numpy as np
import torch
from .data import FEATURES, VERSION, LazySample
from .experiment_data import cohort, chronological_split, eligible_updates, sensor_timelines
from .experiment_metrics import ReferencePredictors, state_metrics
from .experimental_model import ExperimentalModel, configure_stage
from .io import atomic_json
from .provenance import run_metadata
from .train import batch, normalization


from .experiment_config import ExperimentConfig


def model_batch(model, examples, mean, std):
    b = batch([e.sample for e in examples], mean.device, mean, std)
    b['participant_index'] = torch.tensor([model.participant_index.get(e.participant, -1) for e in examples],
                                          device=mean.device)
    return b


def set_counts(model, examples):
    model.answer_counts.zero_()
    for e in examples:
        if e.label.scope in ('current', 'daily'):
            model.answer_counts[model.participant_index[e.participant]] += torch.tensor(
                np.isfinite(e.label.values), device=model.answer_counts.device)


def supervised_loss(model, examples, b):
    outputs = model(b)
    losses = []
    for i, e in enumerate(examples):
        l = e.label
        if l.scope == 'activity':
            losses.append(.2 * torch.nn.functional.cross_entropy(outputs['activity'][i:i+1],
                                                                 torch.tensor([l.activity], device=b['x'].device)))
        else:
            mask = torch.tensor(np.isfinite(l.values), device=b['x'].device)
            if mask.any():
                target = torch.tensor(l.values, device=b['x'].device)
                error = (outputs[l.scope][i, mask] - target[mask]).square().sum()
                penalty = model.bias_penalty(model.participant_index[e.participant], mask)
                losses.append((error + penalty) / mask.sum())
    return torch.stack(losses).mean() if losses else None


def step(model, optimizer, examples, mean, std):
    model.train()
    loss = supervised_loss(model, examples, model_batch(model, examples, mean, std))
    if loss is None or not loss.requires_grad:
        return False
    if not torch.isfinite(loss):
        raise ValueError('Non-finite experimental training loss')
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
    optimizer.step()
    return True


def evaluate(model, examples, mean, std):
    records = []
    model.eval()
    with torch.no_grad():
        for e in examples:
            if e.label.scope in ('current', 'daily'):
                prediction = model(model_batch(model, [e], mean, std))[e.label.scope][0].cpu().numpy()
                records.append((e.participant, e.label, prediction))
    return state_metrics(records)


def fit_supervised(model, train, valid, mean, std, config):
    stages = [('heads' if config.enabled_ssl else 'full', config.epochs, config.learning_rate)]
    if config.upper_finetune:
        stages.append(('upper', config.finetune_epochs, config.finetune_learning_rate))
    best, best_score = None, float('inf')
    history = []
    for stage, epochs, lr in stages:
        configure_stage(model, stage)
        params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([
            dict(params=[p for name, p in params if name != 'personal_bias'], weight_decay=.001),
            dict(params=[p for name, p in params if name == 'personal_bias'], weight_decay=0.)], lr=lr)
        for epoch in range(epochs):
            order = list(train); random.shuffle(order)
            for offset in range(0, len(order), 4):
                step(model, optimizer, order[offset:offset+4], mean, std)
            score = evaluate(model, valid, mean, std)['MAE_macro']
            history.append(dict(stage=stage, epoch=epoch+1, learning_rate=lr, validation_MAE_macro=score))
            if score is not None and score < best_score:
                best_score, best = score, copy.deepcopy(model.state_dict())
        if best is not None:
            model.load_state_dict(best)
    if best is None:
        raise ValueError('No finite validation metric')
    return history


def prequential(model, examples, train, test, mean, std, learning_rate):
    models = {name: copy.deepcopy(model) for name in ('frozen', 'personal_bias', 'full_finetuning')}
    configure_stage(models['frozen'], 'heads')
    for p in models['frozen'].parameters():
        p.requires_grad_(False)
    optimizers = {
        # SGD without momentum/weight decay updates only the observed participant's row.
        'personal_bias': torch.optim.SGD(configure_stage(models['personal_bias'], 'bias'), lr=learning_rate),
        'full_finetuning': torch.optim.SGD(configure_stage(models['full_finetuning'], 'full'), lr=learning_rate),
    }
    used = {e.key for e in train}
    records = {name: [] for name in models}
    predictions = []
    for target in sorted(test, key=lambda e: (e.label.at, e.participant, e.label.id)):
        if target.label.scope not in ('current', 'daily'):
            continue
        new = eligible_updates(examples, used, target.label.at)
        for e in new:
            for name, optimizer in optimizers.items():
                adapted = models[name]
                if e.label.scope in ('current', 'daily'):
                    adapted.answer_counts[adapted.participant_index[e.participant]] += torch.tensor(
                        np.isfinite(e.label.values), device=mean.device)
                step(adapted, optimizer, [e], mean, std)
            used.add(e.key)
        item = dict(participant=target.participant, id=target.label.id, scope=target.label.scope,
                    at=target.label.at.isoformat(), updated_on_labels=len(used)-len(train),
                    latest_update_available=max((e.label.available for e in examples if e.key in used), default=None))
        if item['latest_update_available'] is not None:
            item['latest_update_available'] = item['latest_update_available'].isoformat()
        with torch.no_grad():
            for name, candidate in models.items():
                candidate.eval()
                prediction = candidate(model_batch(candidate, [target], mean, std))[target.label.scope][0].cpu().numpy()
                records[name].append((target.participant, target.label, prediction))
                item[name] = prediction.tolist()
        predictions.append(item)
    return dict(protocol='Predict at question time; update only labels with at AND available strictly earlier; no same-time updates',
                learning_rate=learning_rate, updated_on_labels=len(used)-len(train), predictions=predictions,
                metrics={name: state_metrics(items) for name, items in records.items()})


def run_experiment(rows, directory, config=None):
    config = config or ExperimentConfig()
    config.validate()  # Fail before creating directories or reading any external source.
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'experiment.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(rows, directory, config)


def _run(rows, directory, config):
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    timelines, examples = cohort(rows, text_model=config.text_model)
    from .text import text_metadata
    text_info = text_metadata(next(iter(timelines.values())).encoder) if timelines else text_metadata(None)
    if len(timelines) > 1 and not config.enabled_partial_pooling:
        raise ValueError('Multiple participants require enabled_partial_pooling')
    train, valid, test, boundaries = chronological_split(examples, config.train_before, config.test_from, config.pretrain_before)
    norm_examples = [(e.label, e.sample) for e in train]
    sensors = None
    if config.enabled_ssl:
        from .ssl import ssl_anchors
        sensors = sensor_timelines(timelines)
        plan = [(timeline, t) for timeline in sensors.values()
                for t in ssl_anchors(timeline, boundaries['pretrain_before'], config.horizons,
                                     config.ssl_stride_minutes, config.ssl_max_samples)]
        if not plan:
            raise ValueError('No SSL normalization inputs before cutoff')
        # Generator releases each full hierarchy after normalization; no cohort-sized tensor cache.
        norm_examples = ((None, LazySample(timeline, t)) for timeline, t in plan)
        boundaries['normalization'] = 'SSL input samples strictly before pretrain_before (no future targets)'
    mean_np, std_np = normalization(norm_examples)
    del norm_examples
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean, std = torch.tensor(mean_np, device=device), torch.tensor(std_np, device=device)
    model = ExperimentalModel(participants=sorted(timelines), partial_pooling=config.enabled_partial_pooling,
                              bias_l2=config.bias_l2, hidden=config.hidden).to(device)
    set_counts(model, train)
    ssl_report = None
    if config.enabled_ssl:
        from .ssl import pretrain
        ssl_report = pretrain(model, sensors, boundaries['pretrain_before'], mean, std, batch,
                              epochs=config.ssl_epochs, horizons=config.horizons, mask_probability=config.mask_probability,
                              stride_minutes=config.ssl_stride_minutes, max_samples=config.ssl_max_samples)
    stages = fit_supervised(model, train, valid, mean, std, config)
    report = dict(status='trained', experiment_version=1, production_promoted=False, config=asdict(config),
                  text_encoder=text_info, text_identity=text_info['identity'], text_model=config.text_model,
                  boundaries=boundaries, counts=dict(train=len(train), validation=len(valid), test=len(test)),
                  participants=list(model.participants), ssl=ssl_report, stages=stages,
                  primary_metric='MAE_macro', validation=evaluate(model, valid, mean, std),
                  test=evaluate(model, test, mean, std))
    if config.enabled_evaluation:
        refs = ReferencePredictors(train, examples, ridge=config.enabled_ridge, alpha=config.ridge_alpha)
        report['baselines'] = dict(validation=refs.evaluate(valid), test=refs.evaluate(test))
    if config.enabled_prequential:
        report['prequential'] = prequential(model, examples, train, test, mean, std, config.finetune_learning_rate)
    report['run'] = run_metadata(config.seed, config.epochs, asdict(config),
                                {name: [dict(participant=e.participant, id=e.label.id, at=e.label.at.isoformat(),
                                             available=e.label.available.isoformat()) for e in items]
                                 for name, items in [('train', train), ('validation', valid), ('test', test)]})
    checkpoint = dict(experiment_version=1, settings=model.settings, participants=list(model.participants),
                      partial_pooling=config.enabled_partial_pooling, bias_l2=config.bias_l2,
                      weights=model.state_dict(), mean=mean_np.tolist(), std=std_np.tolist(),
                      features=FEATURES, feature_version=VERSION, report=report)
    # Deliberately incompatible with production inference (no production 'version').
    tmp = directory / 'experiment.tmp'
    torch.save(checkpoint, tmp); os.replace(tmp, directory / 'experiment.pt')
    atomic_json(directory / 'experiment.json', report)
    return report


def load_experiment(path, device='cpu'):
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get('experiment_version') != 1 or state.get('feature_version') != VERSION or state.get('features') != FEATURES:
        raise ValueError('Incompatible experimental checkpoint')
    model = ExperimentalModel(participants=state['participants'], partial_pooling=state['partial_pooling'],
                              bias_l2=state['bias_l2'], **state['settings']).to(device)
    model.load_state_dict(state['weights']); model.eval()
    return model, torch.tensor(state['mean'], device=device), torch.tensor(state['std'], device=device), state['report']
