"""Opt-in shared model with a regularized five-dimensional participant offset."""
import torch
from torch import nn
from .model import HierarchicalLSTM


class ExperimentalModel(HierarchicalLSTM):
    def __init__(self, participants=(), partial_pooling=False, bias_l2=10., **kwargs):
        super().__init__(**kwargs)
        if bias_l2 <= 0:
            raise ValueError('bias_l2 must be positive')
        self.participants = tuple(participants)
        self.participant_index = {p: i for i, p in enumerate(self.participants)}
        self.bias_l2 = float(bias_l2)
        self.personal_bias = nn.Parameter(torch.zeros(len(participants), 5)) if partial_pooling else None
        self.register_buffer('answer_counts', torch.zeros(len(participants), 5))

    def forward(self, b):
        result = super().forward(b)
        if self.personal_bias is not None:
            indices = b['participant_index']
            known = indices >= 0
            offset = result['current'].new_zeros((len(indices), 5))
            offset[known] = self.personal_bias[indices[known]]
            for scope in ('current', 'daily'):
                result[scope] = result[scope] + offset
        return result

    def bias_penalty(self, index, mask):
        if self.personal_bias is None or index < 0:
            return self.current.weight.new_zeros(())
        # Repeated over n observed answers this is lambda * ||b_p||^2.
        # Thus each scale's effective penalty is lambda/n_p: sparse people shrink more.
        return self.bias_l2 * (self.personal_bias[index, mask].square()
                              / self.answer_counts[index, mask].clamp_min(1)).sum()


def configure_stage(model, stage):
    """Head-only excludes personal conditioning and every encoder/fusion parameter."""
    if stage not in ('full', 'heads', 'upper', 'bias'):
        raise ValueError('Unknown training stage')
    heads = ('current.', 'daily.', 'activity.', 'personal_bias')
    upper = ('macro.', 'history_projection.', 'top_day.', 'top_work.')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(stage == 'full' or
                                 (stage in ('heads', 'upper') and name.startswith(heads)) or
                                 (stage == 'upper' and name.startswith(upper)) or
                                 (stage == 'bias' and name == 'personal_bias'))
    return [p for p in model.parameters() if p.requires_grad]
