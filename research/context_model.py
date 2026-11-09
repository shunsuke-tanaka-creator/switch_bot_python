"""Three hierarchical LSTMs, two metric spaces; no GPS or task=place identity."""

import torch
from torch import nn
from torch.nn import functional as fn
from .model import HierarchicalLSTM
from .data import F


class ContextLSTM(nn.Module):
    def __init__(self, hidden=24):
        super().__init__()
        self.encoder = HierarchicalLSTM(hidden=hidden, adaptation=False)
        self.place = nn.Sequential(nn.Linear(hidden, 32), nn.Tanh())
        self.task = nn.Sequential(nn.Linear(hidden, 32), nn.Tanh())
        self.next_observation = nn.Linear(64, F)

    def forward(self, b):
        h = self.encoder(b)["embedding"]
        p = fn.normalize(self.place(h), dim=-1)
        t = fn.normalize(self.task(h), dim=-1)
        return dict(place=p, task=t, next=self.next_observation(torch.cat([p, t], -1)))


def relation_loss(z, ids):
    """Same axis identity is pulled together, different identities separated.
    Includes cross-task positive place pairs and cross-place positive task pairs.
    Unknown/unanswered values contribute no fabricated labels.
    """
    losses = []
    for i in range(len(ids)):
        if ids[i] is None:
            continue
        for j in range(i):
            if ids[j] is None:
                continue
            similarity = (z[i] * z[j]).sum()
            losses.append(
                1 - similarity if ids[i] == ids[j] else fn.relu(similarity - 0.2)
            )
    return torch.stack(losses).mean() if losses else z.sum() * 0
