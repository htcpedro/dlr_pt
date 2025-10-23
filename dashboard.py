import os
import glob
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# =========================================================
# --- Load Latest Forecast Data ---
# =========================================================
FOLDER = "forecasts"

try:
    files = [f for f in glob.glob(os.path.join(FOLDER, "*.csv")) if os.path.isfile(f)]
    forecast_file = max(files, key=os.path.getmtime)
    forecast_ratings = forecast_file.replace(".csv", "_allspans.npz")
except ValueError:
    raise FileNotFoundError("No forecast files found in the 'forecasts' folder.")

# Load data
df_ratings = pd.read_csv(forecast_file)
ratings_all = np.load(forecast_ratings, allow_pickle=True)
df_spans = pd.read_csv("data/spans.csv")

# Extract key data
time = pd.to_datetime(df_ratings["time"])
ratings_forecast = df_ratings["nor"].values
lat_span, lon_span = df_spans["start_la"].tolist(), df_spans["start_lo"].tolist()

# Legend sampling (every 10th span)
lat_legend = lat_span[::10]
lon_legend = lon_span[::10]
span_legend = [f"{i * 10 + 1}" for i in range(len(lat_legend))]

n_spans = df_spans.shape[0]
lat = df_spans["mid_la"]
lon = df_spans["mid_lo"]
rating_nor = ratings_all["nor"]

# Frequency of MLE index occurrences
freq_mle_nonzero = df_ratings["nor_mle"].value_counts()
freq_mle = np.zeros(n_spans)
for i, v in zip(freq_mle_nonzero.index, freq_mle_nonzero.values):
    if i < n_spans:
        freq_mle[i] = v


# =========================================================
# --- Helper Plot Functions ---
# =========================================================
def plot_ts(ts_idx: int) -> go.Figure:
    """Plot the forecast time series."""
    fig = go.Figure()

    # Main line
    fig.add_trace(go.Scattergl(
        x=time, y=ratings_forecast,
        line=dict(shape="hv", color="royalblue"),
        name="NOR", legendgroup="nor"
    ))

    # Highlight selected timestamp
    fig.add_trace(go.Scattergl(
        x=[time[ts_idx]], y=[ratings_forecast[ts_idx]],
        mode="markers",
        marker=dict(symbol="circle", size=16, color="rgba(0,0,0,0)",
                    line=dict(color="red", width=2)),
        name="Selected",
        showlegend=False
    ))

    fig.update_layout(
        yaxis_title="<b>Line Ratings [A]</b>",
        xaxis_title="<b>Time</b>",
        margin=dict(l=0, r=0, t=25, b=0),
        xaxis=dict(range=[time.iloc[0] - pd.Timedelta(hours=1),
                          time.iloc[-1] + pd.Timedelta(hours=1)]),
        yaxis=dict(range=[ratings_forecast.min() * 0.5,
                          ratings_forecast.max() * 1.2])
    )
    return fig


def plot_map(ts_idx: int) -> go.Figure:
    """Plot spans map and highlight MLE."""
    mle_idx = int(df_ratings.loc[ts_idx, "nor_mle"])
    mle_lat = df_spans.loc[mle_idx, "mid_la"]
    mle_lon = df_spans.loc[mle_idx, "mid_lo"]
    mle_rating = df_ratings.loc[ts_idx, "nor"]

    rating = rating_nor[:, ts_idx]

    fig = go.Figure()

    # Line path
    fig.add_trace(go.Scattermap(
        lat=lat_span, lon=lon_span,
        mode="lines", line=dict(width=2, color="black"),
        hoverinfo="skip", showlegend=False
    ))

    # Span points
    fig.add_trace(go.Scattermap(
        lat=lat, lon=lon, mode="markers",
        marker=dict(size=10, color=rating, colorbar=dict(title="NOR (A)")),
        hovertext=[f"{r:.1f}" for r in rating],
        showlegend=False
    ))

    # Span legend labels
    fig.add_trace(go.Scattermap(
        lat=lat_legend, lon=lon_legend,
        mode="markers+text",
        marker=dict(size=10, color="rgba(0,0,0,0)"),
        text=span_legend, textfont=dict(size=12, color="black"),
        textposition="top right",
        showlegend=False
    ))

    # Highlight MLE
    fig.add_trace(go.Scattermap(
        lat=[mle_lat], lon=[mle_lon],
        mode="markers",
        marker=dict(size=28, color="rgba(255,0,0,0.5)"),
        hovertemplate=f"MLE: {mle_rating:.1f} A",
        showlegend=False
    ))

    fig.update_layout(
        map=dict(
            style="carto-positron",
            center=dict(lat=lat.mean(), lon=lon.mean()),
            zoom=9
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig


def plot_bar() -> go.Figure:
    """Bar plot for MLE frequency counts."""
    fig = go.Figure(go.Bar(
        x=list(range(1, n_spans + 1)),
        y=freq_mle,
        marker=dict(color="steelblue"),
        name="MLE counts"
    ))
    fig.update_layout(
        xaxis_title="<b>Span ID</b>",
        yaxis_title="<b>MLE counts</b>",
        margin=dict(l=0, r=0, t=25, b=0)
    )
    return fig


# =========================================================
# --- Dash Layout ---
# =========================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>Forecast Line Ratings</title>
    {%favicon%}
    {%css%}
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-4C0GBEZLEV"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-4C0GBEZLEV');
    </script>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""

app.layout = lambda: html.Div([
    dbc.Container([
        html.H4(
            f"Forecasted Normal Ratings for 150 kV Corgas–Falagueira line from {time.iloc[0]} to {time.iloc[-1]}",
            style={"textAlign": "center", "marginBottom": "1rem"}
        ),

        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='ts-plot',
                    figure=plot_ts(0),
                    style={"height": "40vh"}   # Top graph: 40% of viewport height
                )
            ], md=12)
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    figure=plot_bar(),
                    style={"height": "50vh"}   # Bar chart: 40% height
                )
            ], md=6),

            dbc.Col([
                html.P(
                    "Map showing ratings for all spans. The red circle highlights the most limiting span (MLE) "
                    "for the selected time step in the timeseries plot. Change the timestamp by clicking on the plot "
                    "above.",
                    style={"fontSize": "0.9rem", "marginBottom": "0.5rem"}
                ),
                html.Div(
                    dcc.Graph(id="map", figure=plot_map(0), style={"height": "50vh"}),
                    style={"overflow": "hidden"}  # prevents expanding beyond view
                )
            ], md=6)
        ])
    ],
        fluid=True,
        style={"height": "100vh", "overflow": "hidden"}  # Keep container full screen, no scroll
    )
])


# =========================================================
# --- Callbacks ---
# =========================================================
@app.callback(
    Output("map", "figure"),
    Output("ts-plot", "figure"),
    Input("ts-plot", "clickData"),
    prevent_initial_call=True
)
def update_plots(clickData):
    """Update map and time series when user clicks."""
    if not clickData:
        return plot_map(0), plot_ts(0)

    x_val = pd.Timestamp(clickData["points"][0]["x"], tz="UTC")
    idx = np.argmin(np.abs(x_val - time))
    return plot_map(idx), plot_ts(idx)


server = app.server

# =========================================================
# --- Run ---
# =========================================================
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8050)

