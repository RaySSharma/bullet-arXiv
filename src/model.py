import logging

import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class Vectorizer(BaseEstimator, TransformerMixin):
    """Vectorize data after tokenizing and lemmatizing the input.
    """

    def __init__(
        self,
        vectorizer="tf-idf",
        vectorizer_kwargs={},
        user_stopwords={},
        language="english",
    ):
        """Constructor for pre-processing vectorization.

        Args:
            vectorizer (str, optional): Choice of 'td-idf' for TfidfVectorizer, or 'counts' for CountsVectorizer. Defaults to 'tf-idf'.
            vectorizer_kwargs (dict, optional): Keywords fed into vectorizer. Defaults to {}.
            user_stopwords (dict, optional): User defined stopwords. Defaults to {}.
            language (str, optional): Language of underlying documents. Defaults to "english".
        """
        super().__init__()
        self.vectorizer_kwargs = vectorizer_kwargs
        self.user_stopwords = user_stopwords
        self.language = language
        self.stopwords = self._construct_stopwords()
        self.bow = {}

        if vectorizer == "tf-idf":
            self.vectorizer = TfidfVectorizer
        elif vectorizer == "counts":
            self.vectorizer = CountVectorizer
        else:
            raise NotImplementedError

    def _construct_stopwords(self):
        A = set(nltk.corpus.stopwords.words(self.language))
        B = set(self.user_stopwords)
        stopwords = A.union(B)
        stopwords_str = " ".join(stopwords)
        return self.tokenize_lemmatize(stopwords_str)

    def _tokenize(self, text):
        return nltk.word_tokenize(text, language=self.language)

    def _lemmatize(self, text):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(t) for t in text]
        return " ".join(lemmatized)

    def tokenize_lemmatize(self, X):
        if isinstance(X, str):
            X_ = self._lemmatize(self._tokenize(X)).split()
        else:
            X_ = X.apply(self._tokenize).apply(self._lemmatize)
        return X_

    def _build_bow(self, X):
        bow_text = X.str.split(" ").explode().unique()
        bow = {}
        for value, key in enumerate(bow_text):
            bow[key] = value
        return bow

    def transform(self, X):
        try:
            X_ = self.tokenize_lemmatize(X)
            return self.vectorizer.transform(X_)
        except (AttributeError, TypeError) as err:
            logging.warning(err)
            logging.warning("Fit before transforming.")

    def fit(self, X, y=None):
        X_ = self.tokenize_lemmatize(X)
        self.bow = self._build_bow(X_)
        self.vectorizer = self.vectorizer(
            token_pattern=r"\b\w{3,}\w+\b",
            analyzer="word",
            lowercase=True,
            stop_words=self.stopwords,
            vocabulary=self.bow,
            **self.vectorizer_kwargs,
        )
        self.vectorizer.fit(X_)
        return self


class SparsePCA(PCA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_dense(self, X):
        try:
            X_ = X.toarray()
        except AttributeError:
            X_ = X
        return X_

    def fit(self, X, y=None):
        X_ = self._make_dense(X)
        super().fit(X_, y)

    def transform(self, X, y=None):
        X_ = self._make_dense(X)
        super().transform(X, y)


class LDACluster(BaseEstimator, TransformerMixin):
    def __init__(
        self, n_clusters, lda_kwargs={}, random_state=0,
    ):
        self.n_clusters = n_clusters
        self.lda_kwargs = lda_kwargs
        self.random_state = random_state
        self.labellers = self._setup_labellers()

    def _setup_labellers(self):
        labellers = [
            LatentDirichletAllocation(
                random_state=self.random_state, **self.lda_kwargs,
            )
            for _ in range(self.n_clusters)
        ]
        return labellers

    def fit(self, X, y):
        for i, labeller in enumerate(self.labellers):
            X_cluster = X[y == i]
            labeller.fit(X_cluster)
        return self

    def transform(self, X, y):
        vectorized_data = []
        for i, labeller in enumerate(self.labellers):
            X_cluster = X[y == i]
            X_cluster_label = labeller.transform(X_cluster)
            vectorized_data.append(X_cluster_label)
        return vectorized_data
