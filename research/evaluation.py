"""Pure protocol helpers shared by evaluation tests and the ML runner."""
from .data import dt


def chronological_split(examples):
    """Production day split, including label reception guards (shared with preflight)."""
    dates = sorted({l.at.date() for l, _ in examples if l.scope in ('current', 'daily')})
    if len(dates) < 3:
        return [], [], [], None, None
    a = max(1, int(len(dates) * .6))
    b = max(a + 1, int(len(dates) * .8))
    train_end, test_start = dates[a], dates[min(b, len(dates) - 1)]
    tr = [e for e in examples if e[0].at.date() < train_end and e[0].available.date() < train_end]
    va = [e for e in examples if train_end <= e[0].at.date() < test_start and e[0].available.date() < test_start]
    te = [e for e in examples if e[0].at.date() >= test_start]
    return tr, va, te, train_end, test_start


def baseline_metrics(examples):
    """Score received-cutoff references on the same source-separated validation labels."""
    import numpy as np
    from .data import SCALES
    from .experiment_metrics import sequential_baselines
    errors = {}
    for label, spec in examples:
        if label.scope not in ("current", "daily"):
            continue
        reference = sequential_baselines(spec.timeline.labels, label.at, label.scope)
        for i, name in enumerate(SCALES):
            if not np.isfinite(label.values[i]) or not reference[name]["n"]:
                continue
            key = (label.source, label.scope, name)
            group = errors.setdefault(key, dict(mean=[], last=[], floor=[]))
            for method in ("mean", "last", "floor"):
                prediction = 1. if method == "floor" else reference[name][method]
                group[method].append(abs(prediction - float(label.values[i])))
    return {source: {scope: {name: {
                method: dict(n=len(errors.get((source, scope, name), {}).get(method, [])),
                             mae=float(np.mean(errors[(source, scope, name)][method]))
                             if errors.get((source, scope, name), {}).get(method) else None)
                for method in ("mean", "last", "floor")} for name in SCALES}
                for scope in ("current", "daily")}
            for source in {l.source for l, _ in examples}}


def online_updates(examples, used, cutoff, train_before):
    return sorted(
        [(label, sample) for label, sample in examples
         if label.id not in used and str(label.at.date()) >= str(train_before)
         and label.at < cutoff and label.available < cutoff],
        key=lambda item: (item[0].available, item[0].at, item[0].id),
    )
