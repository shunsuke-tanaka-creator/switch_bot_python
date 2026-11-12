"""Optional frozen local multilingual sentence encoder. Never sends participant text off-host."""
from functools import lru_cache
from pathlib import Path
import hashlib
import numpy as np
from .data import TEXT_DIM


def encoder_identity(directory):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError('text_model must be an existing local model directory')
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if path.is_file() and not any(part.startswith('.') for part in path.relative_to(root).parts):
            digest.update(str(path.relative_to(root)).encode())
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(chunk)
    return digest.hexdigest()


def local_encoder(directory):
    path = str(Path(directory).resolve())
    return _encoder(path, encoder_identity(path))


@lru_cache(maxsize=2)
def _encoder(directory, identity):
    encoder = LocalEncoder(directory)
    if encoder.identity != identity:
        raise ValueError('text encoder changed while loading')
    return encoder


class LocalEncoder:
    def __init__(self,directory):
        root=Path(directory).resolve()
        if not root.is_dir():raise ValueError('text_model must be an existing local model directory')
        from sentence_transformers import SentenceTransformer
        self.model=SentenceTransformer(str(root),device='cpu',local_files_only=True,trust_remote_code=False)
        self.identity=encoder_identity(root)
        dimension = getattr(self.model, 'get_embedding_dimension', None)
        dims = dimension() if dimension is not None else self.model.get_sentence_embedding_dimension()
        self.projection=np.random.default_rng(17).normal(size=(dims,TEXT_DIM)).astype(np.float32)/np.sqrt(TEXT_DIM)

    @lru_cache(maxsize=10000)
    def encode(self,text):
        if not text:return np.zeros(TEXT_DIM,np.float32)
        embedding=self.model.encode(text[-4096:],normalize_embeddings=True,show_progress_bar=False)
        vector=np.asarray(embedding)@self.projection
        return (vector/max(np.linalg.norm(vector),1e-8)).astype(np.float32)
