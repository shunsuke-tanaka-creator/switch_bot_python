"""Dependency-light, disabled-by-default experimental flags."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ExperimentConfig:
    enabled_evaluation: bool = False
    enabled_ssl: bool = False
    enabled_partial_pooling: bool = False
    enabled_prequential: bool = False
    enabled_ridge: bool = False
    upper_finetune: bool = False
    epochs: int = 20
    ssl_epochs: int = 5
    finetune_epochs: int = 5
    hidden: int = 48
    seed: int = 17
    learning_rate: float = .001
    finetune_learning_rate: float = .0001
    bias_l2: float = 10.
    ridge_alpha: float = 1.
    mask_probability: float = .2
    horizons: tuple = (5, 15, 30)
    ssl_stride_minutes: int = 60
    ssl_max_samples: int = 256  # per participant; bounded memory/work
    train_before: str | None = None
    test_from: str | None = None
    pretrain_before: str | None = None

    def validate(self):
        if not any((self.enabled_evaluation, self.enabled_ssl, self.enabled_partial_pooling, self.enabled_prequential)):
            raise ValueError('Experimental features are disabled; explicitly enable at least one')
        if self.enabled_ridge and not self.enabled_evaluation:
            raise ValueError('Ridge requires enabled_evaluation')
        if self.enabled_prequential and not self.enabled_partial_pooling:
            raise ValueError('Three-way prequential evaluation requires enabled_partial_pooling')
        if self.upper_finetune and not self.enabled_ssl:
            raise ValueError('Upper-layer fine-tuning requires enabled_ssl')
        if min(self.epochs, self.ssl_epochs, self.finetune_epochs, self.hidden) < 1:
            raise ValueError('Epoch counts and hidden size must be positive')
        if not 0 < self.finetune_learning_rate < self.learning_rate:
            raise ValueError('Require 0 < finetune_learning_rate < learning_rate')
        if not all(math.isfinite(v) for v in (self.bias_l2, self.ridge_alpha, self.learning_rate,
                                               self.finetune_learning_rate, self.mask_probability)):
            raise ValueError('Hyperparameters must be finite')
        if self.bias_l2 <= 0 or self.ridge_alpha <= 0:
            raise ValueError('Regularization strengths must be positive')
        if not 0 < self.mask_probability <= 1:
            raise ValueError('mask_probability must be in (0, 1]')
        if self.ssl_stride_minutes < 5 or self.ssl_max_samples < 1:
            raise ValueError('Invalid SSL stride or sample limit')
        if not self.horizons or any(h not in (5, 10, 15, 20, 25, 30) for h in self.horizons):
            raise ValueError('SSL horizons must be 5–30 minutes in five-minute steps')
