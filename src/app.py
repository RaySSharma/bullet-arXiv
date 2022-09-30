import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, exceptions
from dash.dependencies import Input, Output

from src import ROOT_DIR

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(__name__, external_stylesheets=external_stylesheets)

DATA = pd.read_hdf(ROOT_DIR / "data/processed/data.hdf5", key="test")
TOTAL_NUM_CLUSTERS = DATA["cluster"].max() + 1


def build_widgets():
    cluster_slider = dcc.Slider(
        value=TOTAL_NUM_CLUSTERS,
        min=0,
        max=TOTAL_NUM_CLUSTERS,
        step=1,
        tooltip={"placement": "bottom", "always_visible": True},
        id="cluster-number",
    )
    slider_label = html.B(
        children=["Cluster #: ", TOTAL_NUM_CLUSTERS], id="slider-label"
    )

    entry_box = dcc.Textarea(
        value="Place your own abstract here.", id="custom-abstract"
    )

    scatter_plot = build_scatter_figure()
    scatter_plot.add_traces(
        go.Scatter(
            x=DATA["tSNE_X"],
            y=DATA["tSNE_Y"],
            mode="markers",
            marker={
                "color": DATA["cluster"],
                "size": 10,
                "colorscale": "Spectral",
                "showscale": False,
                "cmax": TOTAL_NUM_CLUSTERS - 1,
                "cmin": 0,
            },
            customdata=DATA["cluster"],
        )
    )

    scatter_plot = dcc.Graph(figure=scatter_plot, id="scatter-plot")
    keywords_widgets = html.Ul(
        children=DATA["labels"].explode().unique(), id="keywords", style={"columns": 2}
    )
    store = dcc.Store(id="current-cluster", data=TOTAL_NUM_CLUSTERS)

    return [
        cluster_slider,
        slider_label,
        entry_box,
        scatter_plot,
        keywords_widgets,
        store,
    ]


def build_html(widgets):
    (
        cluster_slider,
        slider_label,
        entry_box,
        scatter_plot,
        keywords_widget,
        store,
    ) = widgets

    header = html.H2(children=["Clustering abstracts from arXiv:astro-ph"])
    description = html.H3(children=["<Description>"])
    keywords = html.Div(
        children=["Keywords:", keywords_widget], style={"text-align": "left"}
    )

    row1 = html.Div(
        children=[
            html.Div(
                children=[slider_label, cluster_slider, html.Br(),],
                className="six columns offset-by-one",
                style={"text-align": "center", "justify-content": "center"},
            ),
            html.Div(
                children=[
                    html.Div(children=[scatter_plot], className="eight columns",),
                    html.Div(
                        children=[html.Br(), html.Br(), html.Br(), html.Br(), keywords],
                        className="two columns",
                    ),
                ],
                className="twelve columns",
            ),
        ],
    )
    row2 = html.Div(
        children=[
            html.Div(
                children=[html.H3(["Methods"]), "Methods upon methods"],
                className="three columns",
                id="methods",
            ),
            html.Div(
                children=[html.H3(["Outcomes"]), "Outcomes upon outcomes"],
                className="three columns",
                id="outcomes",
            ),
            html.Div(
                children=[html.H3(["Contact"])], className="three columns", id="contact"
            ),
        ]
    )

    layout = html.Div(
        children=[header, description, row1, row2, store], className="twelve columns"
    )
    return layout


def build_scatter_figure():
    fig = go.Figure()
    fig.update_layout(
        xaxis={"title": None, "range": [-60, 60]},
        yaxis={"title": None, "range": [-60, 60]},
        height=850,
        width=1200,
        showlegend=False,
        margin=go.layout.Margin(l=0, r=0, b=0, t=0,),
    )
    return fig


def plot_scatter(X, marker_kwargs):
    if "color" not in marker_kwargs.keys():
        marker_kwargs["color"] = X["cluster"]
    marker_kwargs["showscale"] = False
    marker_kwargs["cmax"] = TOTAL_NUM_CLUSTERS - 1
    marker_kwargs["cmin"] = 0

    scatter = go.Scatter(
        x=X["tSNE_X"],
        y=X["tSNE_Y"],
        mode="markers",
        marker=marker_kwargs,
        customdata=X["cluster"],
    )
    return scatter


@app.callback(
    Output("slider-label", "children"),
    Output("current-cluster", "data"),
    Input("scatter-plot", "clickData"),
)
def update_cluster_number(clickData):
    if clickData is None:
        return TOTAL_NUM_CLUSTERS
    cluster_number = clickData["points"][0]["customdata"]
    return ["Cluster #: ", cluster_number], cluster_number


@app.callback(
    Output("keywords", "children"), Input("current-cluster", "data"),
)
def update_keywords(cluster_number):
    ix = DATA["cluster"] == cluster_number
    new_keys = DATA.loc[ix, "labels"].explode().unique()
    return [html.Li(x) for x in new_keys]


@app.callback(
    Output("scatter-plot", "figure"), Input("current-cluster", "data"),
)
def update_scatter(cluster_number):
    scatter_plot = build_scatter_figure()

    ix = DATA["cluster"] == cluster_number
    df_cluster = DATA.loc[ix]
    df_not_cluster = DATA.loc[~ix]

    if cluster_number == TOTAL_NUM_CLUSTERS:
        scatter_plot.add_traces(
            plot_scatter(DATA, {"size": 10, "colorscale": "Spectral"})
        )
    else:
        scatter_plot.add_traces(
            plot_scatter(df_not_cluster, {"size": 6, "color": "dimgray",},)
        )
        scatter_plot.add_traces(
            plot_scatter(df_cluster, {"size": 10, "colorscale": "Spectral",},)
        )

    return scatter_plot


if __name__ == "__main__":
    widgets = build_widgets()
    app.layout = build_html(widgets)
    app.run_server(port=8889, debug=True, use_reloader=True)

