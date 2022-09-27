"""Constructors for modeling
"""
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.abstract import FormatText

class Tokenizer(object):
    """Tokenizes and lemmatizes input text while skipping stopwords.
    """

    def __init__(self, user_stopwords=None, language="english"):
        """Tokenizer constructor.

        Args:
            user_stopwords (list, optional): List of user-provided stopwords. Defaults to None.
            language (str, optional): Corpus language. Defaults to "english".
        """
        if user_stopwords is None:
            user_stopwords = []

        self.language = language
        self.stopwords = self.construct_stopwords(user_stopwords)

    def construct_stopwords(self, user_stopwords):
        """Combine user-provided stopwords with nltk stopwords, then tokenize.

        Args:
            user_stopwords (list, optional): List of user-provided stop words. Defaults to None.

        Returns:
            list: Tokenized stopwords.
        """
        base_stopwords = set(nltk.corpus.stopwords.words(self.language))
        extra_stopwords = set(user_stopwords)
        stopwords = base_stopwords.union(extra_stopwords)
        stopwords_str = " ".join(stopwords)
        return self.tokenize(stopwords_str)

    def tokenize(self, text):
        """Tokenize input text

        Args:
            text (str): String to tokenize.

        Returns:
            list: List of tokens within string.
        """
        return nltk.word_tokenize(text, language=self.language)

    def lemmatize(self, tokens):
        """Lemmatize input, assuming it has been pre-tokenized.

        Args:
            tokens (list): List of tokens.

        Returns:
            list: List of lemmatized tokens.
        """
        lemmatizer = nltk.stem.WordNetLemmatizer()
        tokens_lemmatized = []
        for token in tokens:
            if token in self.stopwords:
                lemma = token
            else:
                lemma = lemmatizer.lemmatize(token)
            tokens_lemmatized.append(lemma)
        return tokens_lemmatized

    def __call__(self, text):
        tokens = nltk.word_tokenize(text)
        return self.lemmatize(tokens)


class Vectorizer(BaseEstimator, TransformerMixin):
    """Vectorize data after tokenizing and lemmatizing the input. DEPRECATED.
    """

    def __init__(
        self,
        vectorizer="tf-idf",
        vectorizer_kwargs=None,
        user_stopwords=None,
        language="english",
    ):
        """Constructor for pre-processing vectorization.

        Args:
            vectorizer (str, class; optional): Choice of 'td-idf' or 'counts',
                or user-provided vectorizer. Defaults to 'tf-idf'.
            vectorizer_kwargs (dict, optional): Keywords fed into vectorizer. Defaults to {}.
            user_stopwords (dict, optional): User defined stopwords. Defaults to {}.
            language (str, optional): Language of underlying documents. Defaults to "english".
        """
        super().__init__()
        self.vectorizer_kwargs = vectorizer_kwargs
        self.user_stopwords = user_stopwords
        self.language = language
        self.tokenizer = Tokenizer(self.language)
        self.stopwords = self.construct_stopwords()
        self.bow = {}
        self.estimator = None

        if isinstance(vectorizer, str):
            if vectorizer == "tf-idf":
                self.vectorizer = TfidfVectorizer
            elif vectorizer == "counts":
                self.vectorizer = CountVectorizer
            else:
                raise NotImplementedError
        else:
            self.vectorizer = vectorizer

    def construct_stopwords(self):
        """Combine user-provided stopwords with nltk stopwords, then tokenize.

        Returns:
            list: List of tokenized stopwords.
        """
        base_stopwords = set(nltk.corpus.stopwords.words(self.language))
        user_stopwords = set(self.user_stopwords)
        stopwords = base_stopwords.union(user_stopwords)
        stopwords_str = " ".join(stopwords)
        return self.tokenizer.tokenize(stopwords_str)

    def _build_bow(self, X):
        bow_text = X.str.split(" ").explode().unique()
        bow = {}
        for value, key in enumerate(bow_text):
            bow[key] = value
        return bow

    def transform(self, X):
        """Vectorizes input documents after fitting.

        Args:
            X (array-like): Input documents.

        Returns:
            array-like: Vectorized input documents.
        """
        return self.estimator.transform(X)

    def fit(self, X, y=None):
        """Processes input documents, builds a bag-of-words, then feeds to vectorizer.

        Args:
            X (array-like): Input documents
            y (array-like, optional): Unused, kept for compatibility. Defaults to None.
        """
        self.estimator = self.vectorizer(
            tokenizer=self.tokenizer, analyzer="word", **self.vectorizer_kwargs,
        )
        self.estimator.fit(X)
        self.bow = self.estimator.vocabulary_
        return self


class SparsePCA(PCA):
    """Convenience class to allow PCA to be performed on sparse data by first making it dense.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_dense(self, X):
        try:
            X_ = X.toarray()
        except AttributeError:
            X_ = X
        return X_

    def fit_transform(self, X, y=None):
        X_ = self._make_dense(X)
        return super().fit(X_, y).transform(X_)


class LDACluster(BaseEstimator, TransformerMixin):
    """Performs LatentDirichletAllocation on individual KMeans clusters.
    """

    def __init__(
        self, n_clusters, vectorizer_kwargs, lda_kwargs, random_state=0,
    ):
        self.n_clusters = n_clusters
        self.vectorizer_kwargs = vectorizer_kwargs
        self.lda_kwargs = lda_kwargs
        self.random_state = random_state
        self.labellers = self._setup_labellers()

    def _setup_labellers(self):
        labellers = []
        for _ in range(self.n_clusters):
            formatter = FormatText()
            vectorizer = CountVectorizer(
                random_state=self.random_state, **self.vectorizer_kwargs
            )
            lda = LatentDirichletAllocation(
                random_state=self.random_state, **self.lda_kwargs,
            )
            pipeline = Pipeline([("formatter", formatter), ("vectorizer", vectorizer), ("lda", lda)])
            labellers.append(pipeline)
        return labellers

    def fit(self, X, y):
        """Fits each LDA to its corresponding cluster data.

        Args:
            X (array-like, (N,M)): Input data
            y (array-like, (N,)): K-Means cluster index for each row of X.
        """
        for i, labeller in enumerate(self.labellers):
            X_cluster = X[y == i]
            labeller.fit(X_cluster)
        return self

    def transform(self, X, y):
        """Transforms data in each KMeans cluster with its own LDA transformer.

        Args:
            X (array-like, (N,M)): Input data
            y (array-like, (N,)): K-Means cluster index for each row of X.

        Returns:
            list: List of LDA topics for each cluster.
        """
        vectorized_data = []
        for i, labeller in enumerate(self.labellers):
            X_cluster = X[y == i]
            X_cluster_label = labeller.transform(X_cluster)
            vectorized_data.append(X_cluster_label)
        return vectorized_data
