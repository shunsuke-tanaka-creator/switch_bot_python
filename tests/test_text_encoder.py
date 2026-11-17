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
    assert not first.model.training
    assert all(not p.requires_grad for p in first.model.parameters())

    replacement = package.SentenceTransformer(
        modules=[BoW(['alpha', 'beta'], word_weights={'alpha': 3})], device='cpu')
    replacement.save(str(original), create_model_card=False)
    refreshed = local_encoder(original)
    assert refreshed is not first
    assert refreshed.identity != first.identity
    assert not np.allclose(refreshed.encode('alpha beta'), vector)


def test_sentence_encoder_survives_ssl_and_experiment_checkpoint(tmp_path):
    from sentence_transformers import SentenceTransformer
    try:
        from sentence_transformers.sentence_transformer.modules import BoW
    except ImportError:
        from sentence_transformers.models import BoW
    from research.experiment_data import cohort, sensor_timelines
    from research.experiment_sample import sample_cohort
    from research.experiment_config import ExperimentConfig
    from research.experiments import run_experiment, load_experiment
    from research.data import dt

    directory = tmp_path / 'encoder'
    SentenceTransformer(modules=[BoW(['alpha', 'beta'])], device='cpu').save(
        str(directory), create_model_card=False)
    rows = sample_cohort(5)
    # Synthetic committed text ensures this exercises encoding, not only empty vectors.
    rows.append(dict(participant='sample-a', id='text', kind='session',
                     at='2026-09-07T09:59:00+09:00', payload=dict(
                         started_at='2026-09-07T09:55:00+09:00',
                         ended_at='2026-09-07T09:59:00+09:00',
                         events=[dict(type='commit', time='2026-09-07T09:58:00+09:00',
                                      data=dict(text='alpha beta'))])))
    timelines, _ = cohort(rows, text_model=str(directory))
    sensors = sensor_timelines(timelines)
    assert sensors['sample-a'].encoder is timelines['sample-a'].encoder
    assert not sensors['sample-a'].labels
    left, right = dt('2026-09-07T09:55:00+09:00'), dt('2026-09-07T10:00:00+09:00')
    expected = timelines['sample-a'].encoder.encode('alpha beta')
    np.testing.assert_allclose(sensors['sample-a'].features(left, right)[-TEXT_DIM:], expected)
    config = ExperimentConfig(enabled_ssl=True, enabled_partial_pooling=True,
                              epochs=1, ssl_epochs=1, hidden=4, ssl_max_samples=1,
                              text_model=str(directory))
    report = run_experiment(rows, tmp_path / 'run', config)
    _, _, _, restored = load_experiment(tmp_path / 'run' / 'experiment.pt')
    assert report['text_encoder']['backend'] == 'sentence-transformers'
    assert restored['text_identity'] == timelines['sample-a'].encoder.identity
    assert restored['text_encoder'] == report['text_encoder']


def test_text_cli_defaults_and_explicit_legacy_mode():
    import argparse
    from research.text_config import add_text_arguments, DEFAULT_TEXT_MODEL
    parser = argparse.ArgumentParser()
    add_text_arguments(parser)
    assert parser.parse_args([]).text_model == DEFAULT_TEXT_MODEL
    assert parser.parse_args(['--character-ngram']).text_model is None
    assert parser.parse_args(['--text-model', '/local/model']).text_model == '/local/model'


def test_missing_sentence_model_does_not_fall_back(tmp_path):
    from research.data import Timeline
    with pytest.raises(ValueError, match='existing local model directory'):
        Timeline([], text_model=tmp_path / 'missing')
