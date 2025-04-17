# LSTM Stock Price Predictor - Streamlit Application

## Overview

This project is a web application built with Streamlit that allows users to visualize historical stock data and predict future stock prices using a Long Short-Term Memory (LSTM) neural network. Users can select a stock ticker symbol, specify a date range for historical data, and configure parameters for the LSTM model and prediction horizon.

The application fetches real-time stock data from Yahoo Finance, preprocesses it, trains an LSTM model, evaluates its performance, predicts future prices, and visualizes the results using interactive Plotly charts.

## Features

*   **Stock Data Fetching:** Downloads historical stock data (Open, High, Low, Close, Volume) for a given ticker symbol and date range using the `yfinance` library.
*   **Data Caching:** Uses Streamlit's caching (`@st.cache_data`) to avoid redundant data downloads for the same parameters.
*   **Data Visualization:** Displays historical data through:
    *   A table showing the latest data points.
    *   Interactive Plotly charts:
        *   Closing Price with Trendline and All-Time High/Low (ATH/ATL) markers.
        *   Trading Volume with ATH/ATL markers.
        *   Candlestick chart showing Open, High, Low, Close prices.
*   **LSTM Model Training:**
    *   Preprocesses data (scaling using `MinMaxScaler`, creating sequences).
    *   Builds and compiles a sequential LSTM model using TensorFlow/Keras.
    *   Trains the model on the historical data, showing real-time progress via a Streamlit progress bar and epoch status updates (using a custom Keras callback).
*   **Model Evaluation:**
    *   Evaluates the trained model on a test set.
    *   Displays evaluation metrics:
        *   Mean Squared Error (MSE) on the test set.
        *   R² Score on the test set.
        *   Plots for Training vs. Validation Mean Absolute Error (MAE) and MSE over epochs.
*   **Future Price Prediction:** Predicts stock prices for a user-defined number of future days based on the trained LSTM model.
*   **Prediction Visualization:** Plots the historical true prices, the model's predictions on the test set, and the predicted future prices on a single interactive chart.
*   **User Configuration:** Allows users to configure:
    *   Stock Ticker Symbol
    *   Start and End Dates for historical data
    *   LSTM Lookback Window (Sequence Length)
    *   Prediction Horizon (Number of future days)
    *   Model Training Hyperparameters (Epochs, Batch Size)
*   **Modular Code Structure:** The codebase is organized into modules for better readability, maintainability, and separation of concerns.

## Project Structure

```
StockForecast-Streamlit/
├── streamlit_app.py         # Main entry point for the Streamlit application
├── requirements.txt         # Project dependencies
├── src/                     # Source code directory
│   ├── __init__.py
│   ├── app.py               # Core application logic and Streamlit UI layout
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
    *   It modifies the Python path to ensure the `src` directory is discoverable.
    *   It imports and calls the `run_app` function from `src/app.py`.

2.  **Main Application (`src/app.py`):**
    *   Sets up the Streamlit page configuration (title, layout).
    *   Defines the user interface using Streamlit widgets (sidebar for inputs, main area for outputs).
    *   Collects user inputs: stock symbol, dates, sequence length, future steps, epochs, batch size.
    *   Orchestrates the workflow when the "Run Analysis & Prediction" button is clicked:
        *   Calls `download_stock_data` ([`src/utils/data_loader.py`](/home/chitko/Projects/StockForecast-Streamlit/src/utils/data_loader.py)) to fetch data. Handles potential errors or empty data.
        *   Displays the raw data table.
        *   Calls plotting functions ([`src/utils/plotting.py`](/home/chitko/Projects/StockForecast-Streamlit/src/utils/plotting.py)) like `plot_close_price`, `plot_volume`, `plot_candlestick` to visualize historical data.
        *   Calls `scale_data` and `create_sequences` ([`src/utils/preprocessing.py`](/home/chitko/Projects/StockForecast-Streamlit/src/utils/preprocessing.py)) to prepare data for the LSTM model. It specifically uses the 'Close' price as the target variable.
        *   Splits the data into training and testing sets.
        *   Calls `create_lstm_model` ([`src/models/lstm_model.py`](/home/chitko/Projects/StockForecast-Streamlit/src/models/lstm_model.py)) to build the neural network.
        *   Calls `train_model` ([`src/models/lstm_model.py`](/home/chitko/Projects/StockForecast-Streamlit/src/models/lstm_model.py)), which uses the `CustomProgressBarCallback` ([`src/models/callbacks.py`](/home/chitko/Projects/StockForecast-Streamlit/src/models/callbacks.py)) to show training progress in the UI.
        *   Calls `evaluate_model` ([`src/models/lstm_model.py`](/home/chitko/Projects/StockForecast-Streamlit/src/models/lstm_model.py)) to get performance metrics (MSE, R²) and predictions on the test set.
        *   Calls `plot_evaluation_metrics` ([`src/utils/plotting.py`](/home/chitko/Projects/StockForecast-Streamlit/src/utils/plotting.py)) to display training history and R² score.
        *   Calls `make_future_predictions` ([`src/models/lstm_model.py`](/home/chitko/Projects/StockForecast-Streamlit/src/models/lstm_model.py)) to forecast future prices.
        *   Calls `plot_predictions` ([`src/utils/plotting.py`](/home/chitko/Projects/StockForecast-Streamlit/src/utils/plotting.py)) to visualize true vs. predicted vs. future prices.

3.  **Data Loading (`src/utils/data_loader.py`):**
    *   Uses `yf.download()` to get data.
    *   Includes `@st.cache_data` decorator to speed up subsequent runs with the same inputs.

4.  **Preprocessing (`src/utils/preprocessing.py`):**
    *   `scale_data`: Scales the 'Close' price column between 0 and 1 using `MinMaxScaler`. This is crucial for LSTM performance. It handles the MultiIndex DataFrame structure returned by `yfinance`.
    *   `create_sequences`: Transforms the time series data into input sequences (of length `sequence_length`) and corresponding target labels (the next day's price) suitable for training the LSTM.

5.  **Plotting (`src/utils/plotting.py`):**
    *   Contains functions dedicated to creating different Plotly figures.
    *   These functions take the stock DataFrame (or model results) as input.
    *   They correctly handle the MultiIndex structure from `yfinance` by selecting the appropriate data Series (e.g., `stock_data['Close'].iloc[:, 0]`).
    *   Includes logic to find and display ATH/ATL markers.

6.  **LSTM Model (`src/models/lstm_model.py`):**
    *   `create_lstm_model`: Defines the architecture (multiple LSTM layers with Dropout, followed by a Dense output layer). Compiles the model with 'adam' optimizer and 'mean_squared_error' loss.
    *   `train_model`: Handles the `model.fit()` process, integrating the custom callback for UI updates.
    *   `evaluate_model`: Uses `model.predict()` on the test set, inverse-transforms the scaled predictions back to original price values, and calculates MSE and R² scores.
    *   `make_future_predictions`: Iteratively predicts future steps one by one, using the previous prediction as input for the next step's sequence.

7.  **Callback (`src/models/callbacks.py`):**
    *   `CustomProgressBarCallback`: A TensorFlow Keras callback that updates a Streamlit progress bar and status text during model training (`on_epoch_begin`, `on_epoch_end`, `on_train_end`).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd StockForecast-Streamlit
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
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
    cd /path/to/StockForecast-Streamlit
    ```
2.  **Run the Streamlit application:**
    ```bash
    streamlit run streamlit_app.py
    ```
3.  **Interact with the App:**
    *   The application will open in your web browser.
    *   Use the sidebar on the left to:
        *   Enter a stock ticker symbol (e.g., `AAPL`, `MSFT`, `TSLA`).
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
*   **TensorFlow / Keras:** Deep learning framework used to build and train the LSTM model.
*   **Scikit-learn:** Used for data preprocessing (`MinMaxScaler`, `train_test_split`) and evaluation metrics (`mean_squared_error`, `r2_score`).
*   **Plotly:** Library for creating interactive charts and visualizations.
*   **NumPy:** Numerical operations, especially for handling arrays during preprocessing and modeling.

## Potential Improvements

*   Add more technical indicators (e.g., MACD, RSI, Bollinger Bands) as features to the LSTM model.
*   Allow selection of different model types (e.g., GRU, ARIMA).
*   Implement model saving and loading to avoid retraining.
*   Add more sophisticated error handling and user feedback.
*   Deploy the application to Streamlit Cloud or another hosting platform.
*   Incorporate confidence intervals for predictions.
*   Add unit and integration tests.
