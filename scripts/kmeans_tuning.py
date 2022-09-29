"""Tune the number of clusters hyperparameter for KMeans clustering.
Use both the KMeans elbow plot and maximizing the silhouette score.
"""
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics import silhouette_score
from src.abstract import FormatText
from src.model import Tokenizer

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


def kmeans_pipeline(
    vectorizer_kwargs, svd_kwargs, kmeans_kwargs, random_state=0, **kwargs
):
    """Pipeline for KMeans fitting.

    Args:
        vectorizer_kwargs (dict): Keyword arguments for TfidfVectorizer.
        svd_kwargs (dict): Keyword arguments for TruncatedSVD.
        kmeans_kwargs (dict): Keyword arguments for KMeans.
        random_state (int, optional): Random seed. Defaults to 0.

    Returns:
        sklearn.pipeline.Pipeline: Pipeline object.
    """
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


def calculate_distortion(X, pipeline):
    """Calculate distortion score from KMeans applied to X.

    Args:
        X (array-like): Input data.
        pipeline (sklearn.pipeline.Pipeline): Pipeline object.

    Returns:
        float: Distortion score.
    """
    distances = pipeline["preprocessing"].transform(X)
    cluster_centers = pipeline["kmeans"].cluster_centers_
    distortion = (
        cdist(distances, cluster_centers, "euclidean").min(axis=1).sum()
        / distances.shape[0]
    )
    return distortion


def calculate_silhouette_score(X, pipeline):
    """Calculate silhouette score from KMeans applied to X

    Args:
        X (array-like): Input data.
        pipeline (sklearn.pipeline.Pipeline): Pipeline object.

    Returns:
        float: Silhouette score.
    """
    distances_val = pipeline["preprocessing"].transform(X)
    labels_val = pipeline["kmeans"].predict(distances_val)
    score = silhouette_score(distances_val, labels_val, metric="euclidean")
    return score


def plot_kmeans_elbow(k_range, distortions):
    """Plotting function for distortion elbow diagram of KMeans clusters.

    Args:
        k_range (list): List of possible k values for Kmeans.
        distortions (list): List of pre-calculated cluster distortion values.

    Returns:
        matplotlib.pyplot.Figure: Figure instance.
    """
    f, ax = plt.subplots(1, 1, figsize=(6, 4))

    x_line = [k_range[0], k_range[-1]]
    y_line = [distortions[0], distortions[-1]]

    ax.plot(k_range, distortions, "C0-")
    ax.plot(x_line, y_line, "k--")
    ax.set_xlabel("k")
    ax.set_ylabel("Distortion")
    ax.set_title("KMeans Elbow Plot")
    return f


if __name__ == "__main__":
    X_train = pd.read_csv("data/processed/train.csv")
    X_val = pd.read_csv("data/processed/val.csv")

    k_range = range(5, 100, 10)
    tokenizer = Tokenizer(
        user_stopwords=STOP_WORDS, language="english", token_pattern=r"[a-zA-Z]{3,}",
    )

    scores = []
    distortions = []
    for k in k_range:
        pipeline_k = kmeans_pipeline(
            vectorizer_kwargs={
                "tokenizer": tokenizer,
                "max_df": 0.95,
                "min_df": 3,
                "ngram_range": (1, 1),
                "analyzer": "word",
            },
            svd_kwargs={"n_components": 100},
            kmeans_kwargs={"n_clusters": k},
            random_state=RANDOM_STATE,
            verbose=1,
        )
        pipeline_k.fit(X_train)

        scores.append(calculate_silhouette_score(X_val, pipeline_k))
        distortions.append(calculate_distortion(X_train, pipeline_k))

    print("[k-value, distortion, silhouette score]")
    results = list(zip(k_range, distortions, scores))
    for result in results:
        print(result)

    fig = plot_kmeans_elbow(k_range=k_range, distortions=distortions)
    fig.savefig(
        "figures/kmeans_elbow_plot.png",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

    fig.show()
