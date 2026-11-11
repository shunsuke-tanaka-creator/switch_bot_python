"""Pure protocol helpers shared by evaluation tests and the ML runner."""
from .data import dt


def online_updates(examples, used, cutoff, train_before):
    return sorted(
        [(label, sample) for label, sample in examples
         if label.id not in used and str(label.at.date()) >= str(train_before)
         and label.at < cutoff and label.available < cutoff],
        key=lambda item: (item[0].available, item[0].at, item[0].id),
    )
