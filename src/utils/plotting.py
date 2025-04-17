import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf # For history object type hint

# Renamed function and updated to use 'Close' column
def plot_close_price(stock_data: pd.DataFrame) -> go.Figure:
    """Plots Closing Price with trendline and ATH/ATL."""
    fig = go.Figure()
    # Check if 'Close' is a top-level column
    if 'Close' in stock_data.columns.get_level_values(0):
        # Select the actual Series for the first ticker under 'Close'
        close_data = stock_data['Close'].iloc[:, 0]

        fig.add_trace(go.Scatter(x=close_data.index, y=close_data, mode='lines', name='Close Price',
                                 line=dict(color='lightblue')))
        # Use the Series for trendline
        fig.add_trace(go.Scatter(x=close_data.index, y=close_data.rolling(window=100).mean(),
                                 mode='lines', name='Trendline', line=dict(color='red', width=0.8)))

        # Check if the Series has any non-null data before plotting ATH/ATL
        if not close_data.empty and close_data.notna().any():
            try:
                all_time_high_close = close_data.idxmax()
                all_time_low_close = close_data.idxmin()
                fig.add_trace(go.Scatter(x=[all_time_high_close], y=[close_data.loc[all_time_high_close]],
                                         mode='markers', marker=dict(color='red', size=10),
                                         name='All-Time High Close'))
                fig.add_trace(go.Scatter(x=[all_time_low_close], y=[close_data.loc[all_time_low_close]],
                                         mode='markers', marker=dict(color='green', size=10),
                                         name='All-Time Low Close'))
                fig.update_layout(annotations=[
                    dict(x=all_time_high_close, y=close_data.loc[all_time_high_close],
                         xref="x", yref="y", text="ATH", showarrow=True, arrowhead=2, ax=-30, ay=-40, font=dict(color="red")),
                    dict(x=all_time_low_close, y=close_data.loc[all_time_low_close],
                         xref="x", yref="y", text="ATL", showarrow=True, arrowhead=2, ax=-30, ay=40, font=dict(color="green")),
                ])
            except ValueError as e:
                st.warning(f"Could not determine ATH/ATL for Close price: {e}")

    # Update title and axis label
    fig.update_layout(title=f"Closing Price Visualization",
                      xaxis_title="Date", yaxis_title="Closing Price", showlegend=True)
    return fig

def plot_volume(stock_data: pd.DataFrame) -> go.Figure:
    """Plots Volume with ATH/ATL."""
    fig = go.Figure()
    if 'Volume' in stock_data.columns.get_level_values(0):
        # Select the actual Series for the first ticker under 'Volume'
        volume_data = stock_data['Volume'].iloc[:, 0]

        fig.add_trace(
            go.Bar(x=volume_data.index, y=volume_data, name='Volume', opacity=0.5, marker=dict(color='orange')))

        # Check if the Series has any non-null data before plotting ATH/ATL
        if not volume_data.empty and volume_data.notna().any():
            try:
                all_time_high_volume = volume_data.idxmax()
                all_time_low_volume = volume_data.idxmin()
                fig.add_trace(go.Scatter(x=[all_time_high_volume], y=[volume_data.loc[all_time_high_volume]],
                                         mode='markers', marker=dict(color='red', size=10), name='All-Time High Volume'))
                fig.add_trace(go.Scatter(x=[all_time_low_volume], y=[volume_data.loc[all_time_low_volume]],
                                         mode='markers', marker=dict(color='green', size=10), name='All-Time Low Volume'))
                fig.update_layout(annotations=[
                    dict(x=all_time_high_volume, y=volume_data.loc[all_time_high_volume],
                         xref="x", yref="y", text="ATH", showarrow=True, arrowhead=2, ax=-30, ay=-40, font=dict(color="red")),
                    dict(x=all_time_low_volume, y=volume_data.loc[all_time_low_volume],
                         xref="x", yref="y", text="ATL", showarrow=True, arrowhead=2, ax=-30, ay=40, font=dict(color="green")),
                ])
            except ValueError as e:
                 st.warning(f"Could not determine ATH/ATL for Volume: {e}")

    fig.update_layout(title=f"Volume Visualization",
                      xaxis_title="Date", yaxis_title="Volume", showlegend=True)
    return fig

def plot_candlestick(stock_data: pd.DataFrame) -> go.Figure:
    """Plots Candlestick chart with ATH/ATL."""
    fig = go.Figure()
    required_cols = ['Open', 'High', 'Low', 'Close']
    # Check if all required metrics are in the top level of columns
    if all(col in stock_data.columns.get_level_values(0) for col in required_cols):
        # Select the Series for the first ticker for each required column
        open_data = stock_data['Open'].iloc[:, 0]
        high_data = stock_data['High'].iloc[:, 0]
        low_data = stock_data['Low'].iloc[:, 0]
        close_data = stock_data['Close'].iloc[:, 0]

        fig.add_trace(go.Candlestick(x=stock_data.index, # Index is shared
                                     open=open_data, high=high_data,
                                     low=low_data, close=close_data,
                                     name='Candlestick'))

        # Check if 'Close' Series has non-null data for ATH/ATL
        if not close_data.empty and close_data.notna().any():
            try:
                all_time_high = close_data.idxmax()
                all_time_low = close_data.idxmin()
                fig.add_trace(go.Scatter(x=[all_time_high], y=[close_data.loc[all_time_high]],
                                         mode='markers', marker=dict(color='red', size=10), name='All-Time High Close'))
                fig.add_trace(go.Scatter(x=[all_time_low], y=[close_data.loc[all_time_low]],
                                         mode='markers', marker=dict(color='green', size=10), name='All-Time Low Close'))
                fig.update_layout(annotations=[
                    dict(x=all_time_high, y=close_data.loc[all_time_high],
                         xref="x", yref="y", text="ATH", showarrow=True, arrowhead=2, ax=-30, ay=-40, font=dict(color="red")),
                    dict(x=all_time_low, y=close_data.loc[all_time_low],
                         xref="x", yref="y", text="ATL", showarrow=True, arrowhead=2, ax=-30, ay=40, font=dict(color="green")),
                ])
            except ValueError as e:
                st.warning(f"Could not determine ATH/ATL for Candlestick Close: {e}")

    fig.update_layout(title=f"Candlestick Chart",
                      xaxis_title="Date", yaxis_title="Price", showlegend=True, xaxis_rangeslider_visible=False)
    return fig

def plot_evaluation_metrics(history: tf.keras.callbacks.History, r2: float):
    """Plots training/validation MAE, MSE, and R2 score."""
    st.subheader("LSTM Model Evaluation Results")
    tab1, tab2, tab3 = st.tabs(["MAE", "MSE", "R² Score"])

    with tab1:
        fig_mae = go.Figure()
        fig_mae.add_trace(go.Scatter(x=np.arange(1, len(history.history['mean_absolute_error']) + 1),
                                     y=history.history['mean_absolute_error'], mode='lines', name='Training MAE', line=dict(color='blue')))
        fig_mae.add_trace(go.Scatter(x=np.arange(1, len(history.history['val_mean_absolute_error']) + 1),
                                     y=history.history['val_mean_absolute_error'], mode='lines', name='Validation MAE', line=dict(color='orange')))
        fig_mae.update_layout(title='Training and Validation Mean Absolute Error (MAE)', xaxis_title='Epochs', yaxis_title='MAE', legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig_mae, use_container_width=True)

    with tab2:
        fig_mse = go.Figure()
        fig_mse.add_trace(go.Scatter(x=np.arange(1, len(history.history['mean_squared_error']) + 1),
                                     y=history.history['mean_squared_error'], mode='lines', name='Training MSE', line=dict(color='green')))
        fig_mse.add_trace(go.Scatter(x=np.arange(1, len(history.history['val_mean_squared_error']) + 1),
                                     y=history.history['val_mean_squared_error'], mode='lines', name='Validation MSE', line=dict(color='red')))
        fig_mse.update_layout(title='Training and Validation Mean Squared Error (MSE)', xaxis_title='Epochs', yaxis_title='MSE', legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig_mse, use_container_width=True)

    with tab3:
        st.metric(label="Test Set R² Score", value=f"{r2:.4f}")
        st.info("R² (Coefficient of Determination) measures how well the predictions approximate the real data points. An R² of 1 indicates perfect prediction.")

def plot_predictions(stock_data_index: pd.Index, y_test_original: np.ndarray, predictions: np.ndarray, future_dates: pd.DatetimeIndex, future_predictions: np.ndarray):
    """Plots true prices, test set predictions, and future predictions."""
    st.subheader("LSTM Stock Price Predictions")
    fig_results = go.Figure()

    # True Prices (Test Set)
    fig_results.add_trace(go.Scatter(x=stock_data_index, y=y_test_original.flatten(), # Use provided index directly
                                     mode='lines', name='True Prices (Test Set)', line=dict(color='blue')))

    # Predicted Prices (Test Set)
    fig_results.add_trace(go.Scatter(x=stock_data_index, y=predictions.flatten(), # Use provided index directly
                                     mode='lines', name='Predicted Prices (Test Set)', line=dict(color='red', dash='dot')))

    # Future Predictions
    fig_results.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(),
                                     mode='lines', name='Future Predictions', line=dict(color='green')))

    fig_results.update_layout(title='Stock Price Prediction: True vs. Predicted vs. Future',
                              xaxis_title='Date', yaxis_title='Stock Price',
                              legend=dict(x=0.01, y=0.99, traceorder='normal'))
    st.plotly_chart(fig_results, use_container_width=True)