"""Dependency-light defaults shared by setup and all training CLIs."""
MODEL_ID = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
MODEL_REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
DEFAULT_TEXT_MODEL = 'data/text-models/paraphrase-multilingual-MiniLM-L12-v2'


def add_text_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--text-model', default=DEFAULT_TEXT_MODEL,
                       help='Local SentenceTransformer directory (prepare with python -m research.text_setup)')
    group.add_argument('--character-ngram', dest='text_model', action='store_const', const=None,
                       help='Explicit legacy character n-gram comparison; no sentence model')
