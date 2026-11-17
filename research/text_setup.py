"""Download pinned public weights once; participant text is never used here."""
import argparse
import json
import os
from pathlib import Path
import tempfile

from .text_config import DEFAULT_TEXT_MODEL, MODEL_ID, MODEL_REVISION


def prepare(directory=DEFAULT_TEXT_MODEL):
    from sentence_transformers import SentenceTransformer
    from .text import LocalEncoder
    target = Path(directory).resolve()
    if target.exists():
        raise FileExistsError(f'{target} already exists; use a new directory to preserve model identity')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.sentence-model-', dir=target.parent) as staging:
        root = Path(staging) / 'model'
        model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device='cpu',
                                    trust_remote_code=False)
        model.save(str(root), create_model_card=False)
        (root / 'source.json').write_text(json.dumps(dict(model_id=MODEL_ID, revision=MODEL_REVISION),
                                                   indent=2) + '\n', encoding='utf-8')
        encoder = LocalEncoder(root)  # Check the exact local-only serving loader.
        if encoder.input_dimension != 384:
            raise ValueError('Unexpected pretrained embedding dimension')
        vector = encoder.encode('今日は研究室で論文を書いています。')
        if vector.shape != (32,):
            raise ValueError('Unexpected projected embedding shape')
        os.rename(root, target)
        return dict(status='ready', text_model=str(target), **encoder.metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=DEFAULT_TEXT_MODEL)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
