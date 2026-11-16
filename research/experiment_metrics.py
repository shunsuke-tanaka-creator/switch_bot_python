"""Participant-macro state metrics and chronological reference predictors."""
from collections import defaultdict
import numpy as np
from .data import SCALES, dt, materialize


def sequential_baselines(labels, at, scope):
    """Received-cutoff personal mean and last answer, without cold-start imputation.

    Unlike the experiment's frozen training mean, the delivery mean updates with
    each received answer. Last-value ordering matches ReferencePredictors.
    """
    at = dt(at)
    prior = sorted((l for l in labels if l.scope == scope and l.at < at and l.available < at),
                   key=lambda l: (l.at, l.available, l.id))
    result = {}
    for i, name in enumerate(SCALES):
        values = [float(l.values[i]) for l in prior if np.isfinite(l.values[i])]
        result[name] = dict(n=len(values), mean=float(np.mean(values)) if values else None,
                            last=values[-1] if values else None)
    return result


def state_metrics(records):
    """records: (participant, Label, prediction[5]); absent labels never become zero."""
    records = list(records)

    def summarize(items):
        totals, counts = {}, {}
        for participant, label, prediction in items:
            mask = np.isfinite(label.values)
            if not mask.any():
                continue
            prediction = np.asarray(prediction)
            if prediction.shape != (5,) or not np.isfinite(prediction[mask]).all():
                raise ValueError('State predictions must be five finite values on observed targets')
            totals.setdefault(participant, np.zeros(5))[mask] += np.abs(prediction[mask] - label.values[mask])
            counts.setdefault(participant, np.zeros(5, dtype=int))[:] += mask
        participants = {}
        for participant in sorted(totals):
            per_scale = np.divide(totals[participant], counts[participant],
                                  out=np.full(5, np.nan), where=counts[participant] > 0)
            participants[participant] = dict(
                MAE=float(np.nanmean(per_scale)),
                scales={s: dict(mae=float(per_scale[i]) if np.isfinite(per_scale[i]) else None,
                                n=int(counts[participant][i])) for i, s in enumerate(SCALES)})
        scales = {}
        for s in SCALES:
            present = [v['scales'][s]['mae'] for v in participants.values() if v['scales'][s]['n']]
            scales[s] = dict(MAE_macro=float(np.mean(present)) if present else None,
                             participants=len(present), n=sum(v['scales'][s]['n'] for v in participants.values()))
        return dict(MAE_macro=float(np.mean([v['MAE'] for v in participants.values()])) if participants else None,
                    participants=len(participants), scales=scales, by_participant=participants)

    result = summarize(records)
    result['by_scope'] = {scope: summarize([r for r in records if r[1].scope == scope])
                          for scope in ('current', 'daily')}
    return result


def mean_values(examples, fallback):
    values = np.stack([e.label.values for e in examples]) if examples else np.empty((0, 5))
    counts = np.isfinite(values).sum(0)
    return np.divide(np.nansum(values, axis=0), counts, out=np.array(fallback, dtype=float), where=counts > 0)


def compact_features(example):
    sample = materialize(example.sample)
    d = int(sample['days']) - 1
    u = int(sample['units'][d]) - 1
    t = int(sample['lengths'][d, u]) - 1
    # Only the latest causal feature bin and personal conditioning, not an answer.
    return np.concatenate([sample['x'][d, u, t], sample['personal']]).astype(np.float64)


class ReferencePredictors:
    def __init__(self, train, all_examples, ridge=False, alpha=1.):
        if alpha <= 0:
            raise ValueError('Ridge alpha must be positive')
        self.train = [e for e in train if e.label.scope in ('current', 'daily')]
        self.all = [e for e in all_examples if e.label.scope in ('current', 'daily')]
        self.ridge = {}
        if ridge:
            for scope in ('current', 'daily'):
                items = [e for e in self.train if e.label.scope == scope]
                if not items:
                    continue
                x = np.stack([compact_features(e) for e in items])
                mean, std = x.mean(0), np.maximum(x.std(0), .1)
                x = (x - mean) / std
                targets = np.stack([e.label.values for e in items])
                weights = []
                for i in range(5):
                    mask = np.isfinite(targets[:, i])
                    if not mask.any():
                        weights.append(None)
                        continue
                    xx, yy = x[mask], targets[mask, i]
                    center_x, center_y = xx.mean(0), yy.mean()
                    xc = xx - center_x
                    # Unpenalized intercept; no sklearn dependency or validation fitting.
                    coef = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ (yy - center_y))
                    weights.append((coef, float(center_y - center_x @ coef)))
                self.ridge[scope] = (mean, std, weights)
        self.ridge_enabled = ridge

    def predict(self, target):
        l, p = target.label, target.participant
        scope_train = [e for e in self.train if e.label.scope == l.scope]
        global_mean = mean_values(scope_train, np.full(5, 3.))
        person = [e for e in scope_train if e.participant == p]
        personal_mean = mean_values(person, global_mean)
        hour_mean = mean_values([e for e in person if e.label.at.hour == l.at.hour], personal_mean)
        prior = sorted((e for e in self.all if e.participant == p and e.label.scope == l.scope
                        and e.label.at < l.at and e.label.available < l.at),
                       key=lambda e: (e.label.at, e.label.available, e.label.id))
        last = personal_mean.copy()
        for e in prior:
            mask = np.isfinite(e.label.values)
            last[mask] = e.label.values[mask]
        result = dict(personal_mean=personal_mean, last_answer=last, time_of_day_mean=hour_mean)
        if self.ridge_enabled:
            pred = global_mean.copy()
            if l.scope in self.ridge:
                mean, std, weights = self.ridge[l.scope]
                x = (compact_features(target) - mean) / std
                for i, fitted in enumerate(weights):
                    if fitted is not None:
                        coef, intercept = fitted
                        pred[i] = x @ coef + intercept
            result['ridge'] = pred
        return result

    def evaluate(self, examples):
        records = defaultdict(list)
        for e in examples:
            if e.label.scope in ('current', 'daily'):
                for name, prediction in self.predict(e).items():
                    records[name].append((e.participant, e.label, prediction))
        return {name: state_metrics(items) for name, items in records.items()}
