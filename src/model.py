from .abstract import Abstract

import nltk

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer


class PreProcess(BaseEstimator, TransformerMixin):
    """Pre-process data by tokenizing and lemmatizing the cleaned input.
    """

    def __init__(self, vectorizer_kwargs={}, user_stopwords={}, language="english"):
        """Constructor for pre-processor.

        Args:
            vectorizer_kwargs (dict, optional): Keywords fed into CountsVectorizer. Defaults to {}.
            user_stopwords (dict, optional): User defined stopwords. Defaults to {}.
            language (str, optional): Language of underlying documents. Defaults to "english".
        """
        super().__init__()
        self.vectorizer_kwargs = vectorizer_kwargs
        self.user_stopwords = user_stopwords
        self.language = language
        self.stopwords = self.construct_stopwords()

    def construct_stopwords(self):
        A = set(nltk.corpus.stopwords.words(self.language))
        B = set(self.user_stopwords)
        stopwords = A.union(B)
        return self.lemmatize(stopwords)

    def tokenize(self, text, **kwargs):
        return nltk.word_tokenize(text, **kwargs)

    def lemmatize(self, text):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        return [lemmatizer.lemmatize(t) for t in text]

    def tokenize_lemmatize(self, text, **kwargs):
        tokens = self.tokenize(text, **kwargs)
        lemmas = self.lemmatize(tokens)
        return lemmas

    def preprocessor(self, text):
        return Abstract(text).clean()

    def transform(self, X, y=None):
        X = X.apply(lambda x: Abstract(x).clean().text)
        vectorizer = CountVectorizer(
            tokenizer=self.tokenize_lemmatize,
            lowercase=True,
            analyzer="word",
            stop_words=self.stopwords,
            **self.vectorizer_kwargs
        )
        return vectorizer.fit_transform(X)

    def fit(self, X, y=None):
        return self

