"""Masked observed features and causal 5–30 minute prediction, opt-in only."""
from datetime import timedelta
import numpy as np
import torch
from torch import nn
from .data import BASE, F, TEXT_DIM, LazySample, dt


TIME_FEATURES = {'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'elapsed'}


def observation_mask(values, lengths=None, use_text=True):
    n = len(BASE)
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask[..., :n] = values[..., n:2*n] > 0
    for name in TIME_FEATURES:
        mask[..., BASE.index(name)] = False
    if use_text:
        mask[..., -TEXT_DIM:] = (values[..., n + BASE.index('text_chars')] > 0)[..., None]
    if lengths is not None:
        mask &= (torch.arange(values.shape[-2], device=values.device) < lengths[..., None])[..., None]
    return mask


def corrupt_observed(values, observed, probability, generator=None):
    selected = (torch.rand(values.shape, device=values.device, generator=generator) < probability) & observed
    # Text coordinates form one feature: hide the entire vector to prevent copying
    # a masked coordinate from the remaining normalized embedding coordinates.
    selected[..., -TEXT_DIM:] = selected[..., -TEXT_DIM:].any(-1, keepdim=True) & observed[..., -TEXT_DIM:]
    hidden = values.clone().masked_fill(selected, 0)
    n = len(BASE)
    hidden[..., n:2*n].masked_fill_(selected[..., :n], 0)
    text_hidden = selected[..., -TEXT_DIM:].any(-1)
    hidden[..., BASE.index('text_chars')].masked_fill_(text_hidden, 0)
    hidden[..., n + BASE.index('text_chars')].masked_fill_(text_hidden, 0)
    return hidden, selected


def masked_mse(prediction, target, observed):
    # Index before subtraction: missing NaNs cannot poison loss or gradients.
    return (prediction[observed] - target[observed]).square().mean() if observed.any() else prediction.sum() * 0


class SSLHeads(nn.Module):
    def __init__(self, hidden, horizons):
        super().__init__()
        self.work = nn.Linear(hidden, 12 * F)
        self.history = nn.Linear(hidden, 28 * F)
        self.future = nn.Linear(hidden, len(horizons) * F)
        self.horizons = tuple(horizons)

    def loss(self, model, b, future, future_observed, mask_probability=.2):
        masked = dict(b)
        x_observed = observation_mask(b['x'], b['lengths'], model.settings['use_text'])
        history_observed = observation_mask(b['history'], use_text=model.settings['use_text'])
        masked['x'], xm = corrupt_observed(b['x'], x_observed, mask_probability)
        masked['history'], hm = corrupt_observed(b['history'], history_observed, mask_probability)
        masked['personal'] = torch.zeros_like(b['personal'])
        encoded = model.representations(masked)
        # Each work representation is conditioned on its day and macro context.
        work = encoded['micro'] + encoded['meso'][:, :, None, :] + encoded['macro'][:, None, None, :]
        reconstructed = self.work(work).reshape_as(b['x'])
        history = self.history(encoded['macro']).reshape_as(b['history'])
        mfm = masked_mse(reconstructed, b['x'], xm) + masked_mse(history, b['history'], hm)
        predicted = self.future(encoded['current']).reshape_as(future)
        ffp = masked_mse(predicted, future, future_observed)
        return mfm + ffp, dict(masked_feature_loss=float(mfm.detach()), future_feature_loss=float(ffp.detach()))


def ssl_anchors(timeline, cutoff, horizons, stride_minutes=60, max_samples=256):
    cutoff = dt(cutoff)
    if stride_minutes < 5 or max_samples < 1:
        raise ValueError('SSL stride must be >=5 minutes and max_samples positive')
    if not horizons or any(h < 5 or h > 30 or h % 5 for h in horizons):
        raise ValueError('SSL horizons must be 5, 10, 15, 20, 25 or 30 minutes')
    events = [e for e in timeline.events if e[0] < cutoff and
              dt(e[2].get('_available_at', e[2].get('received_at', e[0]))) < cutoff]
    if not events:
        return []
    start = events[0][0].replace(second=0, microsecond=0)
    start += timedelta(minutes=(-start.minute) % 5)
    last = min(events[-1][0], cutoff - timedelta(minutes=max(horizons)))
    anchors = []
    t = start
    while t <= last and t + timedelta(minutes=max(horizons)) < cutoff:
        anchors.append(t)
        t += timedelta(minutes=stride_minutes)
    # Deterministic coverage across the training period, never validation/test.
    if len(anchors) > max_samples:
        anchors = [anchors[i] for i in np.linspace(0, len(anchors)-1, max_samples, dtype=int)]
    return anchors


def pretrain(model, timelines, cutoff, mean, std, batch_fn, epochs=1,
             horizons=(5, 15, 30), mask_probability=.2, stride_minutes=60, max_samples=256):
    if epochs < 1 or not 0 < mask_probability <= 1:
        raise ValueError('SSL epochs must be positive and mask_probability in (0, 1]')
    device = mean.device
    heads = SSLHeads(model.settings['hidden'], horizons).to(device)
    # No label heads or participant bias are optimized during SSL.
    encoder = [p for name, p in model.named_parameters()
               if not name.startswith(('current.', 'daily.', 'activity.', 'personal.', 'personal_bias'))]
    optimizer = torch.optim.AdamW(encoder + list(heads.parameters()), lr=.001)
    anchors = [(p, t) for p, timeline in timelines.items()
               for t in ssl_anchors(timeline, cutoff, horizons, stride_minutes, max_samples)]
    if not anchors:
        raise ValueError('No SSL samples before pretraining cutoff')
    sums = dict(masked_feature_loss=0., future_feature_loss=0.)
    steps = 0
    model.train()
    for _ in range(epochs):
        for participant, at in anchors:
            timeline = timelines[participant]
            # This sample is discarded after the step: unlabeled grids do not fill LazySample caches.
            sample = LazySample(timeline, at)
            b = batch_fn([sample], device, mean, std)
            raw = torch.tensor(np.stack([timeline.features(at + timedelta(minutes=h-5),
                                                          at + timedelta(minutes=h),
                                                          at + timedelta(minutes=h)) for h in horizons]), device=device)[None]
            observed = observation_mask(raw, use_text=model.settings['use_text'])
            if not observed.any():
                continue
            future = (raw - mean) / std
            loss, values = heads.loss(model, b, future, observed, mask_probability)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite SSL loss')
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder + list(heads.parameters()), 1., error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            for key in sums:
                sums[key] += values[key]
    if not steps:
        raise ValueError('No observed SSL future targets before cutoff')
    return dict(epochs=epochs, samples=len(anchors), steps=steps, horizons_minutes=list(horizons),
                pretrain_before=dt(cutoff).isoformat(), latest_input_at=max(t for _, t in anchors).isoformat(),
                latest_target_end=(max(t for _, t in anchors) + timedelta(minutes=max(horizons))).isoformat(),
                **{k: v/steps for k, v in sums.items()})
