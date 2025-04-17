import streamlit as st
import numpy as np
import pandas as pd
from datetime import date
import logging

# Import refactored components
from src.utils.data_loader import download_stock_data
# Update import for the renamed plotting function
from src.utils.plotting import (
    plot_close_price, plot_volume, plot_candlestick, # Changed here
    plot_evaluation_metrics, plot_predictions
)
from src.utils.preprocessing import scale_data, create_sequences
from src.models.lstm_model import (
    create_lstm_model, train_model, evaluate_model, make_future_predictions
)
from sklearn.model_selection import train_test_split

# Setup basic logging (optional here if set elsewhere, but good practice)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_app():
    # --- Page Config ---
    st.set_page_config(layout="wide", page_title="Stock Price Predictor", page_icon="📈")

    # --- Title and Description ---
    st.title("📈 LSTM Stock Price Predictor")
    st.markdown("""
        Explore historical stock data and predict future prices using an LSTM (Long Short-Term Memory) neural network.
        Use the sidebar to select a stock symbol, date range, and configure the prediction model.
    """)
    st.divider()

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("Data Selection")
        stock_symbol = st.text_input("Stock Symbol", "GOOG").upper()
        start_date = st.date_input("Start Date", date(2015, 1, 1))
        end_date = st.date_input("End Date", date.today()) # Default to today
        st.info(f"Enter a stock symbol (e.g., AAPL, MSFT, GOOG). Data from [Yahoo Finance](https://finance.yahoo.com/).")
        st.divider()

        st.subheader("Prediction Parameters")
        sequence_length = st.slider("Lookback Window (Days)", min_value=10, max_value=120, value=60, step=5,
                                    help="Number of past days' data used to predict the next day.")
        future_steps = st.slider("Prediction Horizon (Days)", min_value=1, max_value=90, value=30, step=1,
                                 help="Number of future days to predict.")
        st.divider()

        st.subheader("LSTM Model Hyperparameters")
        epochs = st.number_input("Epochs", min_value=5, max_value=200, value=50, step=5,
                                 help="Number of training iterations over the entire dataset.")
        batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32,
                                      help="Number of samples processed before the model is updated.")
        st.divider()

        st.warning("Adjusting parameters will trigger data reloading and model retraining.")
        run_analysis = st.button("🚀 Run Analysis & Prediction", type="primary")

    # --- Main Content Area ---
    if not run_analysis:
        st.info("Adjust the settings in the sidebar and click 'Run Analysis & Prediction' to start.")
        return # Stop execution if button not pressed

    # --- Data Loading and Exploration ---
    st.header(f"📊 Data Exploration for {stock_symbol}")
    status_placeholder = st.empty()
    status_placeholder.info(f"Fetching data for {stock_symbol}...")
    stock_data = download_stock_data(stock_symbol, start_date, end_date)

    if stock_data.empty:
        status_placeholder.error("Failed to load data. Please check the stock symbol and date range.")
        st.stop() # Stop if data loading failed

    status_placeholder.success("Data loaded successfully!")
    st.dataframe(stock_data.tail(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Check if 'Close' column exists before plotting
        if 'Close' in stock_data.columns:
            st.plotly_chart(plot_close_price(stock_data), use_container_width=True) # Use renamed function
        else:
            st.warning("Could not plot Close Price: 'Close' column missing.")
        # Check if 'Volume' column exists before plotting
        if 'Volume' in stock_data.columns:
            st.plotly_chart(plot_volume(stock_data), use_container_width=True)
        else:
             st.warning("Could not plot Volume: 'Volume' column missing.")
    with col2:
        # Check if required columns exist for candlestick
        if all(col in stock_data.columns for col in ['Open', 'High', 'Low', 'Close']):
            st.plotly_chart(plot_candlestick(stock_data), use_container_width=True)
        else:
            st.warning("Could not plot Candlestick: Required columns (Open, High, Low, Close) missing.")


    st.divider()

    # --- Data Preprocessing ---
    st.header("⚙️ Model Training & Prediction")
    with st.spinner("Preprocessing data..."):
        # Set target column directly to 'Close'
        target_col = 'Close'

        # Ensure the target column 'Close' exists before proceeding
        if target_col not in stock_data.columns:
            st.error(f"Target column '{target_col}' not found in the downloaded data. Cannot proceed with model training.")
            st.stop()

        # scale_data now defaults to 'Close', but passing explicitly is fine too
        data_scaled, scaler = scale_data(stock_data, target_col)
        X, y = create_sequences(data_scaled, sequence_length)

        if len(X) == 0:
            st.error(f"Not enough data ({len(stock_data)} days) to create sequences with lookback {sequence_length}. Please select a longer date range or shorter lookback.")
            st.stop()

        # Split data
        test_size = 0.2
        if len(X) * test_size < 1: # Ensure at least one sample in test set
             st.warning(f"Dataset size is very small ({len(X)} samples). Test split might be empty or too small. Consider a larger date range.")
             test_size = max(1 / len(X), 0.1) # Adjust test size dynamically or set a minimum

        if len(X) * test_size < batch_size:
             st.warning(f"Test set size ({int(len(X)*test_size)}) is smaller than batch size ({batch_size}). This might affect validation performance. Consider a larger date range or smaller batch size.")

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        except ValueError as e:
            st.error(f"Error during data splitting: {e}. Check data length and test size.")
            st.stop()

        # Reshape for LSTM: [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        st.write(f"Training data shape: {X_train.shape}")
        st.write(f"Test data shape: {X_test.shape}")

    # --- Model Building & Training ---
    input_shape = (sequence_length, 1) # (time steps, features)
    model = create_lstm_model(input_shape)

    history = train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size)

    # --- Model Evaluation ---
    mse, r2, y_test_original, predictions = evaluate_model(model, X_test, y_test, scaler)
    plot_evaluation_metrics(history, r2)
    st.metric(label="Test Set Mean Squared Error (MSE)", value=f"{mse:.4f}")

    st.divider()

    # --- Future Predictions ---
    with st.spinner("Generating future predictions..."):
        # Get the last sequence from the original scaled data
        last_sequence_scaled = data_scaled[-sequence_length:]
        if len(last_sequence_scaled) < sequence_length:
             st.error("Cannot generate future predictions: Not enough historical data for the initial sequence.")
             st.stop()

        future_predictions = make_future_predictions(model, last_sequence_scaled, scaler, future_steps, sequence_length)

        # Create future date index
        last_date = stock_data.index[-1]
        # Ensure future_dates generation is robust
        try:
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps, freq='B') # Use business days 'B'
        except Exception as e:
            st.error(f"Error generating future dates: {e}")
            # Fallback or alternative date generation if needed
            future_dates = pd.to_datetime([last_date + pd.Timedelta(days=i+1) for i in range(future_steps)])


    # --- Plot Results ---
    # Get the correct index for the test set from the original data
    test_index = stock_data.index[-len(y_test_original):]
    plot_predictions(test_index, y_test_original, predictions, future_dates, future_predictions)

# Note: The main execution block is moved to main_runner.py
# if __name__ == "__main__":
#     run_app()
