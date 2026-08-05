import streamlit as st
import pandas as pd
import numpy as np
import datetime
from functools import lru_cache
from typing import Optional, Tuple, Any  # Added Any for history object

# Import constants
from .constants import (
    DEFAULT_SYMBOL,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_FUTURE_STEPS,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    TRAIN_TEST_SPLIT_RATIO,
    LSTM_UNITS,
    VALIDATION_SPLIT,
)

# Import utility functions
from .utils.data_loader import download_stock_data
from .utils.preprocessing import scale_data, create_sequences, inverse_scale_data
from .utils.plotting import (
    plot_raw_data,
    plot_candlestick,
    plot_volume,
    plot_evaluation_metrics,
    plot_predictions,
)

# Import model functions
from .models.lstm_model import (
    create_lstm_model,
    train_model,
    evaluate_model,
    make_future_predictions,
)
from .models.callbacks import CustomProgressBarCallback

# Type alias for Keras History object (replace with actual type if known)
History = Any


@lru_cache(maxsize=1)
def _training_device() -> str:
    """Returns 'GPU' or 'CPU', depending on what TensorFlow can see."""
    import tensorflow as tf

    return "GPU" if tf.config.list_physical_devices("GPU") else "CPU"


def setup_sidebar() -> Tuple[
    str, datetime.date, datetime.date, int, int, int, int, bool
]:
    """
    Sets up the Streamlit sidebar configuration and returns user inputs.

    Returns:
        tuple: Contains stock_symbol, start_date, end_date, sequence_length,
               future_steps, epochs, batch_size, run_button status.
    """
    st.sidebar.header("Configuration")
    st.sidebar.caption(f"Training device: {_training_device()}")
    stock_symbol = st.sidebar.text_input("Stock Symbol", DEFAULT_SYMBOL).upper()
    start_date = st.sidebar.date_input("Start Date", DEFAULT_START_DATE)
    end_date = st.sidebar.date_input("End Date", DEFAULT_END_DATE)

    st.sidebar.subheader("Model Parameters")
    sequence_length = st.sidebar.slider(
        "Lookback Window (Days)", 10, 120, DEFAULT_SEQUENCE_LENGTH, key="seq_len"
    )
    future_steps = st.sidebar.slider(
        "Prediction Horizon (Days)", 5, 90, DEFAULT_FUTURE_STEPS, key="fut_steps"
    )
    epochs = st.sidebar.number_input(
        "Epochs", min_value=1, max_value=500, value=DEFAULT_EPOCHS, key="epochs"
    )
    batch_size = st.sidebar.number_input(
        "Batch Size",
        min_value=8,
        max_value=128,
        value=DEFAULT_BATCH_SIZE,
        step=8,
        key="batch_size",
    )

    run_button = st.sidebar.button("🚀 Run Analysis & Prediction", key="run_button")
    return (
        stock_symbol,
        start_date,
        end_date,
        sequence_length,
        future_steps,
        epochs,
        batch_size,
        run_button,
    )


def load_and_explore_data(
    stock_symbol: str, start_date: datetime.date, end_date: datetime.date
) -> Optional[pd.DataFrame]:
    """
    Loads stock data, validates it, handles potential 'Close' column issues,
    and displays initial data exploration plots. Returns the original DataFrame structure.
    """
    st.header(f"Analysis for {stock_symbol}")
    if start_date >= end_date:
        st.error("Error: End date must fall after start date.")
        return None

    with st.spinner(f"Downloading data for {stock_symbol}..."):
        data = download_stock_data(stock_symbol, start_date, end_date)
        if data is None or data.empty:
            return None

        # --- Column Handling ---
        close_col_data = None
        volume_col_data = None
        display_data = data.copy()  # Copy for display modification

        if isinstance(data.columns, pd.MultiIndex):
            # Find the actual column names (e.g., ('Close', ''), ('Volume', ''))
            close_col_name = next(
                (col for col in data.columns if col[0] == "Close"), None
            )
            volume_col_name = next(
                (col for col in data.columns if col[0] == "Volume"), None
            )
            open_col_name = next(
                (col for col in data.columns if col[0] == "Open"), None
            )
            high_col_name = next(
                (col for col in data.columns if col[0] == "High"), None
            )
            low_col_name = next((col for col in data.columns if col[0] == "Low"), None)

            if close_col_name:
                close_col_data = data[close_col_name]
                if "Close" not in data.columns:
                    data["Close"] = (
                        close_col_data  # Ensure 'Close' exists for later steps
                    )
            else:
                st.error(
                    "Could not find 'Close' price column in the downloaded multi-index data."
                )
                return None

            if volume_col_name:
                volume_col_data = data[volume_col_name]

            # --- Modified Flattening and Selection ---
            # Create a mapping from original multi-index tuple to flattened name
            original_cols = data.columns.values
            flattened_cols = [
                "_".join(filter(None, col)).strip("_") for col in original_cols
            ]
            flattened_map = {
                orig: flat for orig, flat in zip(original_cols, flattened_cols)
            }

            # Assign flattened names to the display copy
            display_data.columns = flattened_cols

            # Identify which *flattened* names correspond to the standard columns we want to show
            cols_to_show_flattened = []
            standard_names_map = {
                "Open": open_col_name,
                "High": high_col_name,
                "Low": low_col_name,
                "Close": close_col_name,
                "Volume": volume_col_name,
            }
            for simple_name, orig_multi_name in standard_names_map.items():
                if orig_multi_name:  # If the original multi-index column existed
                    flattened_name = flattened_map.get(orig_multi_name)
                    if (
                        flattened_name and flattened_name in display_data.columns
                    ):  # Check if it exists after flattening
                        cols_to_show_flattened.append(flattened_name)

            # Select using the identified flattened names
            if cols_to_show_flattened:
                display_data = display_data[cols_to_show_flattened]
            else:
                st.warning(
                    "Could not identify standard columns (Open, High, Low, Close, Volume) after flattening for display."
                )
            # --- End Modified Flattening and Selection ---

        else:  # Single index
            if "Close" in data.columns:
                close_col_data = data["Close"]
            else:
                st.error("Could not find 'Close' price column in the downloaded data.")
                return None

            if "Volume" in data.columns:
                volume_col_data = data["Volume"]

            # Select common columns for display (original names are fine here)
            cols_to_show = [
                col
                for col in ["Open", "High", "Low", "Close", "Volume"]
                if col in data.columns
            ]
            if cols_to_show:
                display_data = display_data[cols_to_show]

        st.success(f"Data downloaded successfully ({len(data)} rows).")
        # --- End Column Handling ---

    # --- Plotting and Display (remains the same) ---
    st.subheader("Data Exploration")
    col1, col2 = st.columns(2)
    with col1:
        if close_col_data is not None:
            st.plotly_chart(plot_raw_data(close_col_data), width="stretch")
        else:
            st.warning("Could not plot Close Price.")
    with col2:
        if volume_col_data is not None:
            st.plotly_chart(plot_volume(volume_col_data), width="stretch")
        else:
            st.warning("Could not plot Volume.")

    st.plotly_chart(plot_candlestick(data), width="stretch")
    st.dataframe(display_data.tail())  # Display the potentially column-filtered data

    # Return the original data structure (with 'Close' potentially added if needed)
    return data


def preprocess_data(
    data: pd.DataFrame, sequence_length: int
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, np.ndarray]]:
    """
    Scales 'Close' price data, creates sequences for LSTM, and splits
    into training and testing sets.

    Args:
        data (pd.DataFrame): DataFrame containing stock data with a 'Close' column.
        sequence_length (int): The lookback window size.

    Returns:
        Optional[tuple]: Contains X_train, X_test, y_train, y_test, scaler, scaled_data,
                         or None if an error occurs or not enough data.
    """
    st.subheader("Data Preprocessing")
    with st.spinner("Preprocessing data..."):
        try:
            close_prices = data["Close"].values.reshape(-1, 1)
            scaled_data, scaler = scale_data(close_prices)
            sequences = create_sequences(scaled_data, sequence_length)

            if sequences is None or len(sequences[0]) == 0:
                st.warning(
                    f"Not enough data ({len(scaled_data)} points) to create sequences with lookback {sequence_length}. Try an earlier start date or shorter lookback."
                )
                return None
            X, y = sequences

            # Split data
            split_index = int(len(X) * TRAIN_TEST_SPLIT_RATIO)
            if split_index == 0 or split_index == len(X):
                st.warning(
                    f"Train/Test split resulted in an empty set (Train: {split_index}, Test: {len(X) - split_index}). Adjust data range or split ratio."
                )
                return None

            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

            # Reshape for LSTM [samples, time steps, features]
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            st.success("Data preprocessed and split.")
            st.text(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
            return (
                X_train,
                X_test,
                y_train,
                y_test,
                scaler,
                scaled_data,
            )  # Return scaled_data as well
        except Exception as e:
            st.error(f"An error occurred during preprocessing: {e}")
            return None


def train_evaluate_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: Any,  # Scaler object (e.g., MinMaxScaler)
    sequence_length: int,
    epochs: int,
    batch_size: int,
) -> Optional[Tuple[Any, History]]:  # Return type hint for Keras model and history
    """
    Trains the LSTM model using the provided data and evaluates its performance
    on the test set (reporting metrics on the original scale).

    Args:
        X_train, y_train: Training data and targets.
        X_test, y_test: Testing data and targets (scaled).
        scaler: The scaler used for preprocessing.
        sequence_length (int): Lookback window size.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        Optional[tuple]: Contains the trained Keras model and the training History object,
                         or None if an error occurs.
    """
    st.subheader("Model Training & Evaluation")
    progress_bar = st.progress(0)
    status_text = st.empty()
    # Use validation split constant
    callback = CustomProgressBarCallback(progress_bar, status_text, epochs)

    with st.spinner("Training LSTM model..."):
        try:
            model = create_lstm_model(
                input_shape=(sequence_length, 1), units=LSTM_UNITS
            )
            history = train_model(
                model,
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=VALIDATION_SPLIT,  # Use constant
                callbacks=[callback],
            )
            st.success("Model training complete.")
        except Exception as e:
            st.error(f"An error occurred during model training: {e}")
            # Optionally log the full traceback here for debugging
            # import traceback
            # st.error(traceback.format_exc())
            return None

    with st.spinner("Evaluating model..."):
        try:
            # Evaluate on original scale using the scaler
            mse, r2 = evaluate_model(model, X_test, y_test, scaler)
            st.metric(
                label="Test Mean Squared Error (MSE)", value=f"{mse:,.2f}"
            )  # Format MSE
            st.metric(label="Test R-squared (R²)", value=f"{r2:.4f}")

            # Plot training history (loss is typically MAE from model compilation)
            st.plotly_chart(plot_evaluation_metrics(history, mse, r2), width="stretch")
            return model, history
        except Exception as e:
            st.error(f"An error occurred during model evaluation: {e}")
            # import traceback
            # st.error(traceback.format_exc())
            return None


def predict_and_visualize(
    model: Any,  # Keras model
    data: pd.DataFrame,  # Original data with DatetimeIndex
    scaled_data: np.ndarray,  # Full scaled 'Close' price data
    scaler: Any,  # Scaler object
    sequence_length: int,
    future_steps: int,
    X_train: np.ndarray,
    X_test: np.ndarray,  # Needed for calculating indices
) -> None:
    """
    Generates predictions on the test set and for future steps,
    then visualizes the actual vs. predicted prices.
    """
    st.subheader("Price Prediction")
    with st.spinner(f"Generating predictions for the next {future_steps} days..."):
        try:
            # 1. Make predictions on the test set
            test_predictions_scaled = model.predict(X_test)
            test_predictions = inverse_scale_data(
                test_predictions_scaled, scaler
            )  # Shape (n, 1)

            # 2. Generate future predictions
            last_sequence = scaled_data[-sequence_length:].reshape(
                1, sequence_length, 1
            )
            future_predictions_scaled = make_future_predictions(
                model, last_sequence, future_steps
            )
            future_predictions = inverse_scale_data(
                future_predictions_scaled, scaler
            )  # Shape (m, 1)

            # --- Explicitly Flatten Arrays ---
            test_predictions_flat = test_predictions.flatten()  # Ensure 1D
            future_predictions_flat = future_predictions.flatten()  # Ensure 1D
            # --- End Flattening ---

            # 3. Prepare data and indices for plotting
            test_actual_start_idx = len(X_train) + sequence_length
            test_actual_end_idx = test_actual_start_idx + len(X_test)
            if test_actual_end_idx > len(data.index):
                test_actual_end_idx = len(data.index)

            test_plot_idx = data.index[test_actual_start_idx:test_actual_end_idx]

            last_date = data.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), periods=future_steps, freq="B"
            )

            # --- Get actual test values and FLATTEN ---
            actual_test_values = (
                data["Close"].iloc[test_actual_start_idx:test_actual_end_idx].values
            )
            actual_test_values_flat = actual_test_values.flatten()  # Flatten here!
            # --- End Flattening ---

            # Create DataFrames for plotting using the flattened arrays
            # Ensure index length matches flattened actual values length
            actual_test_df = pd.DataFrame(
                {"Actual": actual_test_values_flat},
                index=test_plot_idx[: len(actual_test_values_flat)],
            )

            # Ensure index length matches flattened prediction length
            predicted_test_df = pd.DataFrame(
                {"Predicted": test_predictions_flat},
                index=test_plot_idx[: len(test_predictions_flat)],
            )

            future_df = pd.DataFrame(
                {"Future": future_predictions_flat},
                index=future_dates[: len(future_predictions_flat)],
            )  # Match index length

            st.success("Predictions generated.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            import traceback

            st.error(
                traceback.format_exc()
            )  # Show full traceback in Streamlit for debugging
            st.stop()

    # 4. Visualize Predictions (This part should be okay now)
    try:
        st.plotly_chart(
            plot_predictions(
                actual_data=data["Close"],
                actual_test_df=actual_test_df,
                predicted_test_df=predicted_test_df,
                future_df=future_df,
                sequence_length=sequence_length,
            ),
            width="stretch",
        )
        st.subheader(f"Future Predicted Prices (Next {future_steps} Business Days)")
        st.dataframe(future_df)
    except Exception as e:
        st.error(f"An error occurred during plotting predictions: {e}")
        import traceback

        st.error(traceback.format_exc())  # Show full traceback


def run_app():
    """Main function to run the Streamlit application."""
    st.set_page_config(
        layout="wide",
        page_title="LSTM Stock Predictor",
        initial_sidebar_state="expanded",
    )
    st.title("📈 LSTM Stock Price Predictor")

    # Setup sidebar and get parameters
    params = setup_sidebar()
    (
        stock_symbol,
        start_date,
        end_date,
        sequence_length,
        future_steps,
        epochs,
        batch_size,
        run_button,
    ) = params

    if not run_button:
        st.info(
            "Adjust parameters in the sidebar and click '🚀 Run Analysis & Prediction'."
        )
        st.stop()

    # --- Main Workflow ---
    # 1. Load and explore data
    data = load_and_explore_data(stock_symbol, start_date, end_date)
    if data is None:
        st.stop()  # Stop if data loading failed

    # 2. Preprocess data
    preprocess_result = preprocess_data(data, sequence_length)
    if preprocess_result is None:
        st.stop()  # Stop if preprocessing failed
    assert preprocess_result is not None
    X_train, X_test, y_train, y_test, scaler, scaled_data = preprocess_result

    # 3. Train and evaluate model
    train_eval_result = train_evaluate_model(
        X_train, y_train, X_test, y_test, scaler, sequence_length, epochs, batch_size
    )
    if train_eval_result is None:
        st.stop()  # Stop if training/evaluation failed
    assert train_eval_result is not None
    model, history = train_eval_result

    # 4. Predict and visualize
    predict_and_visualize(
        model,
        data,
        scaled_data,
        scaler,
        sequence_length,
        future_steps,
        X_train,
        X_test,  # Pass train/test sets for index calculation
    )

    st.success("Analysis and prediction complete!")
    st.balloons()


# Note: No need for if __name__ == "__main__": here,
# as this module is imported and run_app() is called by streamlit_app.py
