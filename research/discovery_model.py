"""One hierarchy, two open-set metric spaces, and current/daily scale heads.

The legacy classifier remains available in model.py, but is NOT a submodule of
this model. Context task IDs are optional pairwise supervision, not 24 classes.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as fn
from .data import BASE, F, TEXT_DIM
from .model import HierarchicalLSTM

VERSION = "life-discovery-v1"


class DiscoveryModel(nn.Module):
    def __init__(self, hidden: int = 48, embedding_dim: int = 32):
        super().__init__()
        if hidden < 1 or embedding_dim < 2:
            raise ValueError("Invalid model dimensions")
        self.settings = dict(hidden=hidden, embedding_dim=embedding_dim)
        self.encoder = HierarchicalLSTM(hidden=hidden, adaptation=False)
        # representations() does not access these modules when adaptation=False.
        # Remove, rather than merely ignore, the fixed-class/personal modules.
        del self.encoder.activity
        del self.encoder.personal
        self.place = nn.Linear(hidden, embedding_dim)
        self.behavior = nn.Linear(hidden, embedding_dim)
        self.next_observation = nn.Linear(embedding_dim * 2, F)
        self.reconstruction = nn.Linear(hidden, F)

    def forward(self, batch):
        encoded = self.encoder.representations(batch)
        h, day = encoded["current"], encoded["daily"]
        place = fn.normalize(torch.tanh(self.place(h)), dim=-1)
        behavior = fn.normalize(torch.tanh(self.behavior(h)), dim=-1)
        return dict(
            place=place, behavior=behavior,
            current=1 + 4 * torch.sigmoid(self.encoder.current(h)),
            daily=1 + 4 * torch.sigmoid(self.encoder.daily(day)),
            next=self.next_observation(torch.cat((place, behavior), dim=-1)),
            reconstruction=self.reconstruction(h),
        )


def last_observation(batch):
    index = torch.arange(len(batch["x"]), device=batch["x"].device)
    day = batch["days"].long() - 1
    unit = batch["units"][index, day].long() - 1
    step = batch["lengths"][index, day, unit].long() - 1
    return index, day, unit, step


def observation_mask(target):
    n = len(BASE)
    mask = torch.zeros_like(target)
    mask[..., :n] = target[..., n:2*n]
    for i, name in enumerate(BASE):
        if name.endswith(("_sin", "_cos")) or name in (
            "elapsed", "heartbeat_age", "collection_enabled"
        ):
            mask[..., i] = 0
    mask[..., -TEXT_DIM:] = target[..., n + BASE.index("text_chars")].unsqueeze(-1)
    return mask


def masked_input(batch, probability=0.2, generator=None):
    """Mask observed values AND presence; mask text as one block, not 32 labels."""
    if not 0 < probability <= 1:
        raise ValueError("mask probability must be in (0, 1]")
    result = dict(batch, x=batch["x"].clone())
    idx = last_observation(batch)
    target = batch["x"][idx].clone()
    present = observation_mask(target).bool()
    n = len(BASE)
    selected = torch.zeros_like(present)
    selected[:, :n] = present[:, :n] & (
        torch.rand((len(target), n), device=target.device, generator=generator) < probability
    )
    text = present[:, -TEXT_DIM] & (
        torch.rand(len(target), device=target.device, generator=generator) < probability
    )
    selected[:, -TEXT_DIM:] = text.unsqueeze(-1)
    masked = target.masked_fill(selected, 0)
    masked[:, n:2*n] = masked[:, n:2*n].masked_fill(selected[:, :n], 0)
    # text_chars is also a presence indicator of the text block.
    for j in (BASE.index("text_chars"), n + BASE.index("text_chars")):
        masked[text, j] = 0
    result["x"][idx] = masked
    return result, target, selected.to(target.dtype)


def masked_mse(prediction, target, mask):
    valid = mask.bool() & torch.isfinite(target)
    safe = torch.where(valid, target, torch.zeros_like(target))
    return ((prediction - safe).square() * valid).sum() / valid.sum().clamp_min(1)


def identity_loss(vectors, identities):
    """Pairs from independent, explicit context confirmations only."""
    terms = []
    for i in range(len(identities)):
        for j in range(i):
            if identities[i] is None or identities[j] is None:
                continue
            similarity = (vectors[i] * vectors[j]).sum()
            terms.append(1 - similarity if identities[i] == identities[j]
                         else fn.relu(similarity - 0.3))
    return torch.stack(terms).mean() if terms else vectors.sum() * 0


def variance_loss(vectors):
    # Collapse deterrent, not a guarantee that the axes are disentangled.
    if len(vectors) < 2:
        return vectors.sum() * 0
    return fn.relu(0.05 - torch.sqrt(vectors.var(0, unbiased=False) + 1e-4)).mean()
