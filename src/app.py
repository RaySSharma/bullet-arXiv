import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash
from dash.dependencies import Input, Output

from src import ROOT_DIR

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(__name__, external_stylesheets=external_stylesheets)

DATA = pd.read_hdf(ROOT_DIR / "data/processed/data.hdf5", key="test")
TOTAL_NUM_CLUSTERS = DATA["cluster"].max() + 1


def build_widgets():
    cluster_label = html.B(id="cluster-label",)

    scatter_plot = dcc.Graph(id="scatter-plot", clear_on_unhover=True)

    keywords = html.Div(
        children=["Keywords:", html.Plaintext(children="", id="keywords")],
    )
    store1 = dcc.Store(data=TOTAL_NUM_CLUSTERS, id="current-cluster")
    store2 = dcc.Store(data=DATA.to_json(), id="current-data")

    return [
        cluster_label,
        scatter_plot,
        keywords,
        store1,
        store2,
    ]


def build_html(widgets):
    (cluster_label, scatter_plot, keywords, store1, store2) = widgets

    header = html.H2(children=["Clustering abstracts from arXiv:astro-ph"], id="header")
    description = html.H5(
        children=[
            """The rapidly increasing number of research papers published every day makes it \
            difficult to keep track of what is relevant to your own research. Using NLP, \
            clustering, and labelling algorithms, I try to answer the following questions: Can I \
            cluster papers based on their abstracts, and extract meaningful keywords in an \
            automated way?"""
        ],
        id="description",
    )
    methods = (
        html.Div(
            children=[html.H3("Methods"), "Methods upon methods"],
            className="three columns",
            id="methods",
        ),
    )
    outcomes = html.Div(
        children=[html.H3("Outcomes"), "Outcomes upon outcomes"],
        className="three columns",
        id="outcomes",
    )
    contact = html.Div(
        children=[html.H3("Contact")], className="three columns", id="contact"
    )

    row1 = html.Div(
        children=[
            html.Div(
                children=[cluster_label],
                style={"margin-left": "150px", "text-align": "center"},
                className="six columns",
            ),
            html.Br(),
            html.Div(
                children=[keywords],
                style={
                    "margin-left": "30px",
                    "text-align": "left",
                    "margin-bottom": "10px",
                },
            ),
            html.Div(children=[scatter_plot]),
        ],
        className="twelve columns",
    )
    row2 = html.Div(children=[methods, outcomes, contact])

    layout = html.Div(
        children=[header, description, row1, row2, store1, store2],
        className="twelve columns",
    )
    return layout


@app.callback(
    Output("current-cluster", "data"), Input("scatter-plot", "clickData"),
)
def update_current_cluster(hoverData):
    if hoverData is None:
        return TOTAL_NUM_CLUSTERS
    else:
        current_cluster = hoverData["points"][0]["customdata"]
        return current_cluster


@app.callback(Output("cluster-label", "children"), Input("current-cluster", "data"))
def update_cluster_label(current_cluster):
    return "Cluster #: {}".format(current_cluster)


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
        customdata=DATA["cluster"],
        mode="markers",
        marker={
            "size": 10,
            "color": DATA["cluster"],
            "cmin": 0,
            "cmax": TOTAL_NUM_CLUSTERS - 1,
            "showscale": False,
        },
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
    )
    return fig


if __name__ == "__main__":
    widgets = build_widgets()
    app.layout = build_html(widgets)
    app.run_server(port=8889, debug=True, use_reloader=True)

