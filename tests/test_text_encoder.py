"""Exercise local sentence-model loading without downloading weights or user data."""
import shutil

import numpy as np
import pytest

from research.data import TEXT_DIM
from research.text import encoder_identity, local_encoder


def test_local_encoder_relocation_and_cache_invalidation(tmp_path, monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('HF_HUB_DISABLE_TELEMETRY', '1')
    package = pytest.importorskip('sentence_transformers')
    try:
        from sentence_transformers.sentence_transformer.modules import BoW
    except ImportError:  # sentence-transformers 3.x/4.x
        from sentence_transformers.models import BoW

    original = tmp_path / 'original'
    moved = tmp_path / 'moved'
    model = package.SentenceTransformer(modules=[BoW(['alpha', 'beta'])], device='cpu')
    model.save(str(original), create_model_card=False)
    shutil.copytree(original, moved)

    first = local_encoder(original)
    relocated = local_encoder(moved)
    vector = first.encode('alpha beta')
    assert vector.shape == (TEXT_DIM,)
    assert np.isfinite(vector).all()
    np.testing.assert_allclose(np.linalg.norm(vector), 1, atol=1e-6)
    np.testing.assert_array_equal(first.encode(''), np.zeros(TEXT_DIM, np.float32))
    assert encoder_identity(original) == encoder_identity(moved)
    np.testing.assert_array_equal(vector, relocated.encode('alpha beta'))
    assert local_encoder(original) is first

    replacement = package.SentenceTransformer(
        modules=[BoW(['alpha', 'beta'], word_weights={'alpha': 3})], device='cpu')
    replacement.save(str(original), create_model_card=False)
    refreshed = local_encoder(original)
    assert refreshed is not first
    assert refreshed.identity != first.identity
    assert not np.allclose(refreshed.encode('alpha beta'), vector)
