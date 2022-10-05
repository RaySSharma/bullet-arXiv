"""Train and save models and data products. Defines and trains full pre-processing, LSA,
clustering, and labelling pipeline, as well as tSNE for plotting.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from src.abstract import FormatText
from src.model import LDACluster, Tokenizer
from src import ROOT_DIR

RANDOM_STATE = 0

STOP_WORDS = [
    "doi",
    "preprint",
    "copyright",
    "peer",
    "reviewed",
    "org",
    "https",
    "et",
    "al",
    "author",
    "figure",
    "rights",
    "reserved",
    "permission",
    "used",
    "using",
    "arxiv",
    "license",
    "fig",
    "fig.",
    "al.",
    "Elsevier",
    "PMC",
    "CZI",
]


def construct_clustering_pipeline(
    vectorizer_kwargs, svd_kwargs, kmeans_kwargs, random_state=0, **kwargs
):
    formatter = FormatText()
    vectorizer = TfidfVectorizer(**vectorizer_kwargs)
    svd = TruncatedSVD(**svd_kwargs, random_state=random_state)
    normalizer = Normalizer()
    kmeans = KMeans(**kmeans_kwargs, random_state=random_state)
    pipeline = Pipeline(
        [
            (
                "preprocessing",
                Pipeline(
                    [
                        ("formatter", formatter),
                        ("vectorizer", vectorizer),
                        ("svd", svd),
                        ("normalizer", normalizer),
                    ]
                ),
            ),
            ("kmeans", kmeans),
        ],
        **kwargs
    )

    return pipeline


def fit_pipelines(
    X,
    vectorizer_kwargs,
    svd_kwargs,
    kmeans_kwargs,
    tsne_kwargs,
    lda_kwargs,
    random_state,
):
    n_clusters = kmeans_kwargs.get("n_clusters")

    clustering_pipeline = construct_clustering_pipeline(
        vectorizer_kwargs=vectorizer_kwargs,
        svd_kwargs=svd_kwargs,
        kmeans_kwargs=kmeans_kwargs,
        random_state=random_state,
    )
    tsne = TSNE(**tsne_kwargs, random_state=random_state)
    cluster_labeller = LDACluster(
        n_clusters=n_clusters,
        vectorizer_kwargs=vectorizer_kwargs,
        lda_kwargs=lda_kwargs,
        random_state=random_state,
    )
    clustering_pipeline.fit(X)
    tsne.fit(clustering_pipeline.transform(X))
    cluster_labeller.fit(X, clustering_pipeline.predict(X))
    return clustering_pipeline, tsne, cluster_labeller


def get_cluster_labels(words_per_topic, cluster_labeller):
    def _get_lda_labels(labeller, words_per_topic):
        feature_names = labeller["vectorizer"].get_feature_names_out()
        cluster_keywords = [
            _get_top_n_words(words_per_topic, topic, feature_names)
            for topic in labeller["lda"].components_
        ]
        return np.unique(cluster_keywords)

    def _get_top_n_words(n, topic, feature_names):
        return [feature_names[i] for i in topic.argsort()[: -n - 1 : -1]]

    lda_labels = [
        _get_lda_labels(labeller, words_per_topic)
        for labeller in cluster_labeller.labellers
    ]
    return lda_labels


def construct_df(X, clustering_pipeline, tsne):
    df = X.copy()

    X_preprocessed = clustering_pipeline["preprocessing"].transform(X)
    X_tsne = tsne.fit_transform(X_preprocessed)
    kmeans_labels = clustering_pipeline.predict(X)
    text = clustering_pipeline["preprocessing"]["formatter"].transform(X)

    df["tSNE_X"] = X_tsne[:, 0]
    df["tSNE_Y"] = X_tsne[:, 1]
    df["cluster"] = kmeans_labels
    df["processed_text"] = text
    df["authors"] = df["authors"].apply(lambda x: pd.eval(x, engine="python")).str[0]
    df["labels"] = ", ".join([lda_labels[int(cluster)] for cluster in kmeans_labels])
    return df


if __name__ == "__main__":

    X_train = pd.read_csv(ROOT_DIR / "data/processed/train.csv")
    X_val = pd.read_csv(ROOT_DIR / "data/processed/val.csv")
    X_test = pd.read_csv(ROOT_DIR / "data/processed/test.csv")

    tokenizer = Tokenizer(
        user_stopwords=STOP_WORDS, language="english", token_pattern=r"[a-zA-Z]{3,}"
    )

    clustering_pipeline, tsne, cluster_labeller = fit_pipelines(
        X_train,
        vectorizer_kwargs={
            "tokenizer": tokenizer,
            "max_df": 0.95,
            "min_df": 3,
            "ngram_range": (1, 1),
            "analyzer": "word",
        },
        svd_kwargs={"n_components": 100},
        kmeans_kwargs={"n_clusters": 50},
        tsne_kwargs={"perplexity": 50, "init": "pca", "learning_rate": 200},
        lda_kwargs={"n_components": 5, "learning_method": "online"},
        random_state=RANDOM_STATE,
    )
    lda_labels = get_cluster_labels(
        words_per_topic=5, cluster_labeller=cluster_labeller,
    )

    if not os.path.isdir("data/models"):
        os.path.mkdir("data/models")

    joblib.dump(clustering_pipeline, ROOT_DIR / "data/models/clustering_pipeline.pkl")
    joblib.dump(cluster_labeller, ROOT_DIR / "data/models/cluster_labeller.pkl")

    keys = ["train", "val", "test"]
    for i, X in enumerate([X_train, X_val, X_test]):
        df = construct_df(X, clustering_pipeline, tsne)
        df.to_hdf(ROOT_DIR / "data/processed/data.hdf5", key=keys[i])
