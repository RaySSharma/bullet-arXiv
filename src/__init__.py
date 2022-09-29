import pathlib

import nltk

from . import abstract, model

try:
    nltk.corpus.stopwords.words()
except LookupError:
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
