# LSTM Stock Price Predictor - Streamlit Application

## Overview

This project is a web application built with Streamlit that allows users to visualize historical stock data and predict future stock prices using a Long Short-Term Memory (LSTM) neural network. Users can select a stock ticker symbol, specify a date range for historical data, and configure parameters for the LSTM model and prediction horizon.

The application fetches historical stock data from Yahoo Finance, preprocesses it (focusing on the 'Close' price), trains an LSTM model using TensorFlow/Keras, evaluates its performance, predicts future prices, and visualizes the results using interactive Plotly charts.

## Features

*   **Stock Data Fetching:** Downloads historical stock data (Open, High, Low, Close, Volume) for a given ticker symbol and date range using the `yfinance` library.
*   **Data Caching:** Uses Streamlit's caching (`@st.cache_data`) to avoid redundant data downloads for the same parameters.
*   **Data Validation & Handling:** Checks for valid date ranges, handles empty or NaN data, and correctly processes potential MultiIndex columns returned by `yfinance`.
*   **Data Visualization:** Displays historical data through:
    *   A table showing the latest data points (selected columns).
    *   Interactive Plotly charts:
        *   Closing Price over time.
        *   Trading Volume over time.
        *   Candlestick chart showing Open, High, Low, Close prices.
*   **LSTM Model Training:**
    *   Preprocesses 'Close' price data (scaling using `MinMaxScaler`, creating lookback sequences).
    *   Splits data into training and testing sets based on a defined ratio.
    *   Builds and compiles an LSTM model using the Keras Functional API (Input, LSTM, Dropout, Dense layers).
    *   Trains the model on the historical data, showing real-time progress via a Streamlit progress bar and epoch status updates (using a custom Keras callback). Includes Early Stopping.
*   **Model Evaluation:**
    *   Evaluates the trained model on the test set, calculating metrics on the *original price scale*.
    *   Displays evaluation metrics:
        *   Mean Squared Error (MSE) on the test set.
        *   R-squared (R²) Score on the test set.
    *   Plots the Training vs. Validation Loss (Mean Absolute Error) over epochs.
*   **Future Price Prediction:** Predicts stock prices for a user-defined number of future business days based on the trained LSTM model, using the last sequence of data.
*   **Prediction Visualization:** Plots the actual test prices, the model's predictions on the test set, and the predicted future prices on a single interactive chart. Displays future predicted prices in a table.
*   **User Configuration:** Allows users to configure via the sidebar:
    *   Stock Ticker Symbol
    *   Start and End Dates for historical data
    *   LSTM Lookback Window (Sequence Length)
    *   Prediction Horizon (Number of future days)
    *   Model Training Hyperparameters (Epochs, Batch Size)
*   **Modular Code Structure:** The codebase is organized into modules for better readability, maintainability, and separation of concerns (constants, data loading, preprocessing, plotting, model definition, callbacks, main app logic).

## Project Structure

```
lstm-trend/
├── streamlit_app.py         # Main entry point for the Streamlit application
├── requirements.txt         # Project dependencies
├── src/                     # Source code directory
│   ├── __init__.py
│   ├── app.py               # Core application logic and Streamlit UI layout
│   ├── constants.py         # Defines constants (defaults, model params, etc.)
│   ├── utils/               # Utility functions
│   │   ├── __init__.py
│   │   ├── data_loader.py   # Handles fetching data from yfinance
│   │   ├── plotting.py      # Generates all Plotly visualizations
│   │   └── preprocessing.py # Handles data scaling and sequence creation
│   └── models/              # Machine learning model related code
│       ├── __init__.py
│       ├── lstm_model.py    # Defines, trains, evaluates, and uses the LSTM model
│       └── callbacks.py     # Custom Keras callback for Streamlit progress bar
└── README.md                # This file
```

## How It Works

1.  **Entry Point (`streamlit_app.py`):**
    *   This is the file executed by the `streamlit run` command.
    *   It modifies the Python path (`sys.path`) to ensure the `src` directory is discoverable.
    *   It imports and calls the `run_app` function from [`src/app.py`](/home/chitko/Projects/lstm-trend/src/app.py).

2.  **Main Application (`src/app.py` - `run_app` function):**
    *   Sets up the Streamlit page configuration (title, layout).
    *   Calls `setup_sidebar` to define the user interface using Streamlit widgets in the sidebar and collect user inputs (stock symbol, dates, sequence length, future steps, epochs, batch size).
    *   Waits for the "Run Analysis & Prediction" button click.
    *   **If button clicked:** Orchestrates the main workflow by calling sequential functions:
        *   `load_and_explore_data`:
            *   Calls [`download_stock_data`](/home/chitko/Projects/lstm-trend/src/utils/data_loader.py#L9) ([`src/utils/data_loader.py`](/home/chitko/Projects/lstm-trend/src/utils/data_loader.py)) to fetch data (using caching).
            *   Validates dates and handles potential `yfinance` MultiIndex columns, ensuring 'Close' and 'Volume' are accessible.
            *   Displays a spinner, success/error messages, and the tail of the data.
            *   Calls plotting functions ([`src/utils/plotting.py`](/home/chitko/Projects/lstm-trend/src/utils/plotting.py)) like [`plot_raw_data`](/home/chitko/Projects/lstm-trend/src/utils/plotting.py#L25), [`plot_volume`](/home/chitko/Projects/lstm-trend/src/utils/plotting.py#L41), [`plot_candlestick`](/home/chitko/Projects/lstm-trend/src/utils/plotting.py#L57) to visualize historical data.
            *   Returns the downloaded data or `None` on failure.
        *   `preprocess_data`:
            *   Extracts the 'Close' price column.
            *   Calls [`scale_data`](/home/chitko/Projects/lstm-trend/src/utils/preprocessing.py#L5) ([`src/utils/preprocessing.py`](/home/chitko/Projects/lstm-trend/src/utils/preprocessing.py)) to scale prices between 0 and 1.
            *   Calls [`create_sequences`](/home/chitko/Projects/lstm-trend/src/utils/preprocessing.py#L24) ([`src/utils/preprocessing.py`](/home/chitko/Projects/lstm-trend/src/utils/preprocessing.py)) to create input sequences (X) and target values (y).
            *   Splits sequences into training and testing sets based on [`TRAIN_TEST_SPLIT_RATIO`](/home/chitko/Projects/lstm-trend/src/constants.py#L14) from [`src/constants.py`](/home/chitko/Projects/lstm-trend/src/constants.py).
            *   Reshapes X data for LSTM input `[samples, time_steps, features]`.
            *   Returns split data, scaler, and full scaled data, or `None` on failure.
        *   `train_evaluate_model`:
            *   Sets up Streamlit progress bar and status text elements.
            *   Instantiates [`CustomProgressBarCallback`](/home/chitko/Projects/lstm-trend/src/models/callbacks.py#L5) ([`src/models/callbacks.py`](/home/chitko/Projects/lstm-trend/src/models/callbacks.py)).
            *   Calls [`create_lstm_model`](/home/chitko/Projects/lstm-trend/src/models/lstm_model.py#L13) ([`src/models/lstm_model.py`](/home/chitko/Projects/lstm-trend/src/models/lstm_model.py)) to build the Keras Functional API model.
            *   Calls [`train_model`](/home/chitko/Projects/lstm-trend/src/models/lstm_model.py#L44) ([`src/models/lstm_model.py`](/home/chitko/Projects/lstm-trend/src/models/lstm_model.py)), passing the custom callback and using [`VALIDATION_SPLIT`](/home/chitko/Projects/lstm-trend/src/constants.py#L16) from [`src/constants.py`](/home/chitko/Projects/lstm-trend/src/constants.py).
            *   Calls [`evaluate_model`](/home/chitko/Projects/lstm-trend/src/models/lstm_model.py#L83) ([`src/models/lstm_model.py`](/home/chitko/Projects/lstm-trend/src/models/lstm_model.py)) to get performance metrics (MSE, R²) calculated on the *original* price scale.
            *   Displays metrics using `st.metric`.
            *   Calls [`plot_evaluation_metrics`](/home/chitko/Projects/lstm-trend/src/utils/plotting.py#L77) ([`src/utils/plotting.py`](/home/chitko/Projects/lstm-trend/src/utils/plotting.py)) to display training history (MAE loss) and test metrics.
            *   Returns the trained model and history object, or `None` on failure.
        *   `predict_and_visualize`:
            *   Predicts on the test set (`X_test`) using `model.predict`.
            *   Calls [`make_future_predictions`](/home/chitko/Projects/lstm-trend/src/models/lstm_model.py#L116) ([`src/models/lstm_model.py`](/home/chitko/Projects/lstm-trend/src/models/lstm_model.py)) to forecast future prices iteratively.
            *   Calls [`inverse_scale_data`](/home/chitko/Projects/lstm-trend/src/utils/preprocessing.py#L14) ([`src/utils/preprocessing.py`](/home/chitko/Projects/lstm-trend/src/utils/preprocessing.py)) to convert scaled predictions back to original prices.
            *   Calculates correct date indices for test and future predictions.
            *   Creates Pandas DataFrames for actual test, predicted test, and future predicted values.
            *   Calls [`plot_predictions`](/home/chitko/Projects/lstm-trend/src/utils/plotting.py#L111) ([`src/utils/plotting.py`](/home/chitko/Projects/lstm-trend/src/utils/plotting.py)) to visualize actual vs. predicted vs. future prices.
            *   Displays the future predictions DataFrame.
        *   Displays a success message and balloons.
    *   **If button not clicked:** Shows an informational message.

3.  **Constants (`src/constants.py`):**
    *   Defines default UI values (symbol, dates, model parameters) and model configuration constants (split ratio, LSTM units, validation split, plot colors) using `typing.Final`.

4.  **Data Loading (`src/utils/data_loader.py`):**
    *   Uses `yf.download()` to get data.
    *   Includes `@st.cache_data` decorator to speed up subsequent runs with the same inputs.
    *   Performs basic cleaning (`dropna`).

5.  **Preprocessing (`src/utils/preprocessing.py`):**
    *   `scale_data`: Scales the 'Close' price column between 0 and 1 using `MinMaxScaler`. Returns scaled data and the scaler object.
    *   `inverse_scale_data`: Reverses the scaling using a provided scaler object.
    *   `create_sequences`: Transforms the scaled time series data into input sequences (of length `sequence_length`) and corresponding target labels (the next day's price) suitable for training the LSTM. Returns `None` if data is insufficient.

6.  **Plotting (`src/utils/plotting.py`):**
    *   Contains functions dedicated to creating different Plotly figures (`plot_raw_data`, `plot_volume`, `plot_candlestick`, `plot_evaluation_metrics`, `plot_predictions`).
    *   Uses a helper `_create_layout` for consistent styling (dark theme).
    *   Handles potential MultiIndex columns from `yfinance` when plotting.

7.  **LSTM Model (`src/models/lstm_model.py`):**
    *   `create_lstm_model`: Defines the architecture using Keras Functional API (Input -> LSTM -> Dropout -> LSTM -> Dropout -> Dense -> Dense Output). Compiles the model with 'adam' optimizer and 'mean_absolute_error' loss.
    *   `train_model`: Handles the `model.fit()` process, integrating callbacks (including default `EarlyStopping`). Sets `verbose=0` to rely on the custom callback for UI updates.
    *   `evaluate_model`: Uses `model.predict()` on the test set, inverse-transforms the scaled predictions and actual test values back to original price values using the provided scaler, and calculates MSE and R² scores.
    *   `make_future_predictions`: Iteratively predicts future steps one by one, using the previous prediction as input for the next step's sequence. Returns scaled predictions.

8.  **Callback (`src/models/callbacks.py`):**
    *   `CustomProgressBarCallback`: A TensorFlow Keras callback that updates a Streamlit progress bar (`st.progress`) and status text (`st.empty`) during model training (`on_epoch_begin`, `on_epoch_end`, `on_train_end`).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url> # Replace with your repo URL
    cd lstm-trend
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: TensorFlow installation might require specific steps depending on your system (CPU/GPU). Refer to the official TensorFlow installation guide if needed.*

## Usage

1.  **Navigate to the project directory:**
    ```bash
    cd /path/to/lstm-trend
    ```
2.  **Ensure your virtual environment is activated:**
    ```bash
    source venv/bin/activate # Or equivalent for your OS
    ```
3.  **Run the Streamlit application:**
    ```bash
    streamlit run streamlit_app.py
    ```
4.  **Interact with the App:**
    *   The application will open in your web browser.
    *   Use the sidebar on the left to:
        *   Enter a stock ticker symbol (e.g., `AAPL`, `MSFT`, `GOOGL`).
        *   Select the start and end dates for the historical data.
        *   Adjust the Lookback Window (sequence length) and Prediction Horizon.
        *   Modify the training Epochs and Batch Size.
    *   Click the "🚀 Run Analysis & Prediction" button.
    *   Observe the data loading, historical charts, model training progress, evaluation metrics, and the final prediction chart in the main area of the application.

## Key Technologies

*   **Python:** Core programming language.
*   **Streamlit:** Framework for building the interactive web application UI.
*   **Pandas:** Data manipulation and analysis, handling the stock data DataFrame.
*   **yfinance:** Library to download historical stock market data from Yahoo Finance.
*   **TensorFlow / Keras:** Deep learning framework used to build and train the LSTM model (Functional API).
*   **Scikit-learn:** Used for data preprocessing (`MinMaxScaler`) and evaluation metrics (`mean_squared_error`, `r2_score`).
*   **Plotly:** Library for creating interactive charts and visualizations.
*   **NumPy:** Numerical operations, especially for handling arrays during preprocessing and modeling.

## Potential Improvements

*   Add more technical indicators (e.g., MACD, RSI, Bollinger Bands) as features to the LSTM model.
*   Implement model saving and loading to avoid retraining on subsequent runs.
*   Add more sophisticated error handling and user feedback (e.g., specific yfinance errors).
*   Deploy the application to Streamlit Community Cloud or another hosting platform.
*   Incorporate confidence intervals for predictions.
*   Allow user selection of the target column (e.g., 'Open', 'High') instead of hardcoding 'Close'.
*   Optimize hyperparameters.