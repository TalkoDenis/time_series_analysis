import plotly.graph_objects as go


def visualize_data(forecast_df, train_df):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=train_df["ds"],
            y=train_df["y"],
            mode="lines",
            line=dict(color="red", width=2),
            name="Train data",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df["ds"],
            y=forecast_df["Real_data"],
            mode="lines",
            line=dict(color="red", width=2),
            name="Real data",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df["ds"],
            y=forecast_df["Forecust_data"],
            mode="lines",
            line=dict(color="blue", width=2),
            name="Forecast",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df["ds"],
            y=forecast_df["max"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0,100,80,0.2)",
            name="Forecast interval",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["ds"],
            y=forecast_df["min"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0,100,80,0.2)",
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Our data",
        xaxis_title="Week",
        yaxis_title="N",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )

    fig.show()
