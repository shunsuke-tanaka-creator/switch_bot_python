"""Separate opt-in CLI: experiment artifacts cannot replace serving models."""
from dataclasses import fields
from pathlib import Path
from .experiment_config import ExperimentConfig


def add_parser(sub):
    p = sub.add_parser('experiment', help='Offline opt-in evaluation, SSL and partial pooling')
    from .text_config import add_text_arguments
    add_text_arguments(p)
    p.add_argument('--models', default='data/experiments')
    p.add_argument('--participants', nargs='+', help='Explicit cohort IDs; never auto-enumerates the database')
    p.add_argument('--sample-fixture', action='store_true', help='Use repository artificial cohort; never opens a DB')
    for name in ('evaluation', 'ssl', 'partial-pooling', 'prequential', 'ridge'):
        p.add_argument('--enable-'+name, dest='enabled_'+name.replace('-', '_'), action='store_true')
    p.add_argument('--upper-finetune', action='store_true')
    for name in ('epochs', 'ssl_epochs', 'finetune_epochs', 'hidden', 'seed', 'ssl_stride_minutes', 'ssl_max_samples'):
        p.add_argument('--'+name.replace('_', '-'), type=int, default=getattr(ExperimentConfig(), name))
    for name in ('learning_rate', 'finetune_learning_rate', 'bias_l2', 'ridge_alpha', 'mask_probability'):
        p.add_argument('--'+name.replace('_', '-'), type=float, default=getattr(ExperimentConfig(), name))
    for name in ('train_before', 'test_from', 'pretrain_before'):
        p.add_argument('--'+name.replace('_', '-'), help='Timezone-aware ISO timestamp; exclusive boundary')
    p.add_argument('--horizons', type=int, nargs='+', default=[5, 15, 30])


def run(args):
    config = ExperimentConfig(**{field.name: tuple(args.horizons) if field.name == 'horizons'
                                 else getattr(args, field.name) for field in fields(ExperimentConfig)})
    config.validate()
    if args.sample_fixture:
        if args.participants:
            raise ValueError('sample-fixture cannot be combined with participants')
        from .experiment_sample import sample_cohort
        rows = sample_cohort()
    else:
        if not args.participants:
            raise ValueError('Specify participants or sample-fixture')
        from .service import identifier
        from lifelog_server import Store
        for participant in args.participants:
            identifier(participant)
        if not Path(args.db).is_file():
            raise ValueError('Experiment database must already exist')
        # Work on an in-memory snapshot. Experiment reads never migrate/write the source DB.
        import sqlite3
        store = Store(':memory:')
        try:
            with sqlite3.connect(Path(args.db).resolve().as_uri() + '?mode=ro', uri=True) as source:
                source.backup(store.db)
            rows = [row for p in dict.fromkeys(args.participants) for row in store.rows(p)]
        finally:
            store.close()
    from .experiments import run_experiment
    report = run_experiment(rows, args.models, config)
    return dict(status=report['status'], report=str(Path(args.models)/'experiment.json'),
                checkpoint=str(Path(args.models)/'experiment.pt'), production_promoted=False,
                test_MAE_macro=report['test']['MAE_macro'])
