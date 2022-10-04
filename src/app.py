"""Generate a dashboard to show off bullet_arxiv results. Illustrates results from models applied to
the test data including an interactive plot, listed keywords, methodology, and conclusions.
"""
import string
import textwrap

import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash
from dash.dependencies import Input, Output

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(__name__, external_stylesheets=external_stylesheets)

DATA = pd.read_hdf("data/processed/data.hdf5", key="test").reset_index(drop=True)
TOTAL_NUM_CLUSTERS = DATA["cluster"].max() + 1


def build_hover_data():
    title = (
        DATA["title"]
        .str.title()
        .str.replace(" +", " ", regex=True)
        .apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))
        .apply(lambda x: "<br>".join(textwrap.wrap(x, width=50)))
    )
    authors = (
        DATA["authors"]
        .apply(lambda x: x[1:-1].split(", ")[0])
        .str.translate(str.maketrans("", "", string.punctuation))
        + " et al"
    )
    doi = DATA["doi"]
    arxiv_id = DATA["id"]
    arxiv_sections = DATA["categories"]
    lda_labels = (
        DATA["labels"]
        .apply(", ".join)
        .apply(lambda x: "<br>".join(textwrap.wrap(x, width=50)))
    )
    cluster = DATA["cluster"]
    return pd.DataFrame(
        np.asarray(
            [title, authors, doi, arxiv_id, arxiv_sections, lda_labels, cluster]
        ).T
    )


def build_widgets():
    cluster_label = html.B(id="cluster-label")

    scatter_plot = dcc.Graph(id="scatter-plot", clear_on_unhover=True)

    keywords = html.Div(
        children=[html.B("Keywords:"), html.P(children="", id="keywords")],
    )
    store1 = dcc.Store(data=TOTAL_NUM_CLUSTERS, id="current-cluster")
    store2 = dcc.Store(id="current-data")

    return [
        cluster_label,
        scatter_plot,
        keywords,
        store1,
        store2,
    ]


def build_html(widgets):
    (cluster_label, scatter_plot, keywords, store1, store2) = widgets

    header = html.H2(children=["Clustering abstracts from arXiv"], id="header")
    description = html.H6(
        children=[
            html.P(
                children=[
                    """The rapidly growing number of research papers published every day makes it \
                     difficult to keep track of which papers are even relevant. I scrape 19,177 \
                     abstracts from peer-reviewed publications on the arXiv and use NLP techniques \
                     to answer the following questions:""",
                    html.Ul(
                        [
                            html.Li(
                                """Can papers be reasonably clustered based on information from \
                                 their abstracts?"""
                            ),
                            html.Li(
                                "Can meaningful keywords be generated for these clusters (e.g, \
                                 can I identify sub-fields from the keywords)?"
                            ),
                        ]
                    ),
                ]
            )
        ],
        id="description",
    )
    results = html.Div(
        children=[
            html.H6("Conclusions"),
            html.P(
                children=[
                    html.P(
                        "It is clear that paper abstracts contain \
                        enough information to tightly cluster the papers! However, there are no \
                        metrics that will tell me whether the keywords are entirely meaningful. \
                        Fortunately, from my own knowledge of the domain, I can see the clusterer \
                        does a reasonably good job of identifying sub-fields within astronomy. \
                        For example, Cluster 5 lists `accretion`, `bubble`, `cluster`, `galaxy`, \
                        `gas`, `heating`, and `radio` as keywords -- strong indicators that the \
                        papers are discussing black hole-induced gas heating in galaxies and \
                        galaxy clusters! \
                        "
                    ),
                    html.P(
                        "It is important to note that the methods used here are field-agnostic, \
                         meaning this package can be used to cluster and identify key-words from \
                         papers from any field."
                    ),
                ]
            ),
        ],
        id="outcomes",
    )
    methods = html.Div(
        children=[
            html.H6("Methodology"),
            html.P(
                children=[
                    "I follow a similar metholodogy to ",
                    html.A(
                        "Eren et al (2020)",
                        href="https://doi.org/10.1145/3395027.3419591",
                    ),
                    ". The steps in the NLP pipeline are summarized as follows:",
                ]
            ),
            html.Ul(
                children=[
                    html.Li(
                        """Separate the raw data into training (60%), validation (20%), \
                            and test (20%) sets. I selected papers with DOI numbers only."""
                    ),
                    html.Li(
                        """Format the incoming paper abstracts, stripping LaTeX \
                            equations, digits, and punctuation. Generate lemmatized tokens, \
                            comparing against a set of stop-words."""
                    ),
                    html.Li(
                        """Vectorize text using a TF-IDF transformer, down-weighting tokens \
                            that appear frequently throughout the corpus. The output is a sparse \
                            matrix. A bag-of-words is generated from the training data."""
                    ),
                    html.Li(
                        """Reduce the dimensionality of the sparse, vectorized text using \
                            Latent Semantic Analysis with n=100 components and normalizing the \
                            results."""
                    ),
                    html.Li(
                        """Run K-means clustering, where the number of clusters were chosen by \
                            maximizing the silhouette score of the validation set. Silhouette \
                            scores measure both the distance between points within a \
                            cluster, and the distance between clusters."""
                    ),
                    html.Li(
                        """On each cluster, run Latent Dirichlet Allocation to identify topics \
                            and identify descriptive key words from the bag-of-words."""
                    ),
                    html.Li(
                        """Run t-SNE to flatten the vectorized data down into \
                            2 dimensions for visualization. Points are colored by their K-means \
                            cluster designation. Shown here are papers from the test set."""
                    ),
                ]
            ),
        ],
        id="methods",
    )
    contact = html.Div(
        children=[
            html.H6("Contact"),
            html.B("Primary author: "),
            "Ray Sharma",
            html.Br(),
            html.B("Github: "),
            html.A(
                "https://github.com/RaySSharma/bullet-arXiv",
                href="https://github.com/RaySSharma/bullet-arXiv",
            ),
        ],
        id="contact",
    )

    row1 = html.Div(
        children=[header, description, html.Hr()], className="twelve columns"
    )
    row2 = html.Div(
        children=[
            html.Div(
                children=[
                    html.Div(children=[cluster_label], style={"text-align": "center",}),
                    html.Div(
                        children=[keywords],
                        style={
                            "margin-left": "30px",
                            "justify-content": "left",
                            "text-align": "left",
                            "margin-bottom": "10px",
                        },
                    ),
                    html.Div(children=[html.Div(children=[scatter_plot])]),
                ],
                className="eight columns",
            ),
            html.Div(children=[methods, results, contact], className="four columns"),
        ],
        className="twelve columns",
    )

    layout = html.Div(children=[row1, row2, store1, store2], style={"padding": "20px"})
    return layout


@app.callback(
    Output("current-cluster", "data"), Input("scatter-plot", "clickData"),
)
def update_current_cluster(hoverData):
    if hoverData is None:
        return TOTAL_NUM_CLUSTERS
    else:
        current_cluster = hoverData["points"][0]["customdata"][6]
        return current_cluster


@app.callback(Output("cluster-label", "children"), Input("current-cluster", "data"))
def update_cluster_label(current_cluster):
    if current_cluster == TOTAL_NUM_CLUSTERS:
        cluster_label = "All"
    else:
        cluster_label = current_cluster
    return "Cluster #: {}".format(cluster_label)


@app.callback(
    Output("current-data", "data"), Input("current-cluster", "data"),
)
def update_data(current_cluster):
    if current_cluster == TOTAL_NUM_CLUSTERS:
        ix = DATA["cluster"] < current_cluster
    else:
        ix = DATA["cluster"] == current_cluster
    return ix.to_json()


@app.callback(
    Output("keywords", "children"), Input("current-data", "data"),
)
def update_keywords(current_data):
    X = DATA.loc[pd.read_json(current_data, typ="series")]

    if TOTAL_NUM_CLUSTERS in X["cluster"]:
        new_keywords = "Click on a data point to see the keywords"
    else:
        new_keywords = ", ".join(X["labels"].explode().unique())
    return new_keywords


@app.callback(
    Output("scatter-plot", "figure"), Input("current-data", "data"),
)
def update_scatter_plot(current_data):
    ix = pd.read_json(current_data, typ="series")

    fig = go.Figure()
    scatter = go.Scatter(
        x=DATA["tSNE_X"],
        y=DATA["tSNE_Y"],
        customdata=hover_data,
        mode="markers",
        marker={
            "size": 10,
            "color": DATA["cluster"],
            "cmin": 0,
            "cmax": TOTAL_NUM_CLUSTERS - 1,
            "showscale": False,
        },
        hovertemplate="<br>".join(
            [
                "<b><i>%{customdata[0]}</i></b>",
                "<b>Authors</b>: %{customdata[1]}",
                "<b>DOI</b>: %{customdata[2]}",
                "<b>ArXiv ID</b>: %{customdata[3]}",
                "<b>ArXiv Sections</b>: %{customdata[4]}",
                "<b>LDA Labels</b>: %{customdata[5]}",
                "<b>Cluster #</b>: %{customdata[6]}",
                "<extra></extra>",
            ]
        ),
        selectedpoints=ix[ix].index.values,
    )

    fig.add_traces(scatter)
    fig.update_layout(
        xaxis={"title": None, "range": [-60, 60]},
        yaxis={"title": None, "range": [-60, 60]},
        height=850,
        width=1200,
        showlegend=False,
        margin=go.layout.Margin(l=0, r=0, b=0, t=0),
        hoverlabel={"namelength": -1},
    )
    return fig


widgets = build_widgets()
hover_data = build_hover_data()
app.layout = build_html(widgets)

application = app.server
app.title = "Bullet ArXiv"

if __name__ == "__main__":
    application.run(port=8080, debug=True)
