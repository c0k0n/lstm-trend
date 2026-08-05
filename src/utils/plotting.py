import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Any  # For Keras History object

# Import constants for styling (optional, but good practice)
from ..constants import PLOT_BGCOLOR, PLOT_FONT_COLOR, PLOT_GRID_COLOR


def _create_layout(
    title: str, xaxis_title: str = "Date", yaxis_title: str = "Price USD ($)"
) -> go.Layout:
    """Helper function to create a standard Plotly layout."""
    return go.Layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",  # Use a dark theme
        paper_bgcolor=PLOT_BGCOLOR,
        plot_bgcolor=PLOT_BGCOLOR,
        font=dict(color=PLOT_FONT_COLOR),
        xaxis=dict(gridcolor=PLOT_GRID_COLOR),
        yaxis=dict(gridcolor=PLOT_GRID_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )


def plot_raw_data(close_data: pd.Series) -> go.Figure:
    """
    Plots the raw closing price over time. Accepts a pandas Series.

    Args:
        close_data (pd.Series): Series with DatetimeIndex and closing prices.

    Returns:
        go.Figure: Plotly figure object.
    """
    fig = go.Figure()
    # Use .values to ensure numpy array is passed if needed by Plotly version
    fig.add_trace(
        go.Scatter(
            x=close_data.index, y=close_data.values, mode="lines", name="Close Price"
        )
    )
    fig.update_layout(_create_layout(title="Stock Closing Price"))
    return fig


def plot_volume(volume_data: pd.Series) -> go.Figure:
    """
    Plots the trading volume over time. Accepts a pandas Series.

    Args:
        volume_data (pd.Series): Series with DatetimeIndex and volume data.

    Returns:
        go.Figure: Plotly figure object.
    """
    fig = go.Figure()
    # Use .values to ensure numpy array is passed if needed by Plotly version
    fig.add_trace(go.Bar(x=volume_data.index, y=volume_data.values, name="Volume"))
    fig.update_layout(_create_layout(title="Trading Volume", yaxis_title="Volume"))
    return fig


def plot_candlestick(data: pd.DataFrame) -> go.Figure:
    """
    Creates a candlestick chart of the stock data.

    Args:
        data (pd.DataFrame): DataFrame with DatetimeIndex and OHLC columns.

    Returns:
        go.Figure: Plotly figure object.
    """
    # Handle potential MultiIndex columns from yfinance
    ohlc_cols = {}
    required = ["Open", "High", "Low", "Close"]
    if isinstance(data.columns, pd.MultiIndex):
        for col_name in required:
            # Find the column where the first level matches
            match = [col for col in data.columns if col[0] == col_name]
            if match:
                ohlc_cols[col_name] = data[match[0]]
            else:
                # Cannot create candlestick without all OHLC
                return go.Figure(
                    layout=_create_layout(title=f"Error: '{col_name}' column not found")
                )
    else:
        # Standard columns
        if all(col in data.columns for col in required):
            ohlc_cols = {col: data[col] for col in required}
        else:
            missing = [col for col in required if col not in data.columns]
            return go.Figure(
                layout=_create_layout(title=f"Error: Missing columns {missing}")
            )

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data.index,
                open=ohlc_cols["Open"],
                high=ohlc_cols["High"],
                low=ohlc_cols["Low"],
                close=ohlc_cols["Close"],
                name="Candlestick",
            )
        ]
    )
    fig.update_layout(_create_layout(title="Candlestick Chart"))
    return fig


def plot_evaluation_metrics(history: Any, mse: float, r2: float) -> go.Figure:
    """
    Plots the model training history (loss, val_loss) and displays evaluation metrics.

    Args:
        history (Any): Keras History object returned by model.fit().
        mse (float): Mean Squared Error on the test set (original scale).
        r2 (float): R-squared score on the test set (original scale).

    Returns:
        go.Figure: Plotly figure object.
    """
    fig = make_subplots(rows=1, cols=1)  # Simple plot for loss

    # Check if history object and history attribute exist
    if history and hasattr(history, "history"):
        hist_dict = history.history
        # Check for common loss keys ('loss', 'mae') and validation counterparts
        loss_key = (
            "loss" if "loss" in hist_dict else ("mae" if "mae" in hist_dict else None)
        )
        val_loss_key = (
            "val_loss"
            if "val_loss" in hist_dict
            else ("val_mae" if "val_mae" in hist_dict else None)
        )

        if loss_key:
            fig.add_trace(
                go.Scatter(
                    y=hist_dict[loss_key], mode="lines", name="Training Loss (MAE)"
                ),
                row=1,
                col=1,
            )
        if val_loss_key:
            fig.add_trace(
                go.Scatter(
                    y=hist_dict[val_loss_key],
                    mode="lines",
                    name="Validation Loss (MAE)",
                ),
                row=1,
                col=1,
            )

    # Update layout
    title = f"Model Training History<br>Test MSE: {mse:,.2f} | Test R²: {r2:.4f}"
    fig.update_layout(
        _create_layout(
            title=title, xaxis_title="Epoch", yaxis_title="Mean Absolute Error (Loss)"
        )
    )
    fig.update_layout(height=400)  # Adjust height if needed
    return fig


def plot_predictions(
    actual_data: pd.Series,  # Full actual 'Close' price series
    actual_test_df: pd.DataFrame,  # Actual values for the test period
    predicted_test_df: pd.DataFrame,  # Predicted values for the test period
    future_df: pd.DataFrame,  # Predicted future values
    sequence_length: int,  # To know where training data ends visually
) -> go.Figure:
    """
    Plots the actual stock prices, test set predictions, and future predictions.

    Args:
        actual_data (pd.Series): The complete actual 'Close' price data with DatetimeIndex.
        actual_test_df (pd.DataFrame): DataFrame with 'Actual' column and DatetimeIndex for test period.
        predicted_test_df (pd.DataFrame): DataFrame with 'Predicted' column and DatetimeIndex for test period.
        future_df (pd.DataFrame): DataFrame with 'Future' column and DatetimeIndex for future period.
        sequence_length (int): The lookback window size.

    Returns:
        go.Figure: Plotly figure object.
    """
    fig = go.Figure()

    # Plot Actual Test Data - Use .values
    fig.add_trace(
        go.Scatter(
            x=actual_test_df.index,
            y=actual_test_df["Actual"].values,
            mode="lines",
            name="Actual Test Price",
            line=dict(color="orange"),
        )
    )

    # Plot Predicted Test Data - Use .values
    fig.add_trace(
        go.Scatter(
            x=predicted_test_df.index,
            y=predicted_test_df["Predicted"].values,
            mode="lines",
            name="Predicted Test Price",
            line=dict(color="yellow", dash="dot"),
        )
    )

    # Plot Future Predictions - Use .values
    fig.add_trace(
        go.Scatter(
            x=future_df.index,
            y=future_df["Future"].values,
            mode="lines",
            name="Future Predictions",
            line=dict(color="red", dash="dash"),
        )
    )

    # Determine the overall range for the plot
    # Show some history before test starts + future predictions
    plot_start_date = actual_data.index.min()  # Start from the beginning of loaded data
    if not actual_test_df.empty:
        # If test data exists, maybe start plot a bit before it
        plot_start_date = max(
            plot_start_date, actual_test_df.index.min() - pd.Timedelta(days=90)
        )

    plot_end_date = (
        future_df.index.max() if not future_df.empty else actual_test_df.index.max()
    )

    fig.update_layout(
        _create_layout(title="Actual vs. Predicted Stock Prices"),
        xaxis_range=[plot_start_date, plot_end_date],  # Set x-axis range based on dates
    )
    return fig
