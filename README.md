# StockForecast-Streamlit

Welcome to the StockForecast-Streamlit repository! This repository contains a Streamlit web application for stock price prediction using an LSTM (Long Short-Term Memory) model. The application allows users to explore historical stock data, visualize various stock metrics, and predict future stock prices.

## Features

* **Stock Data Downloading** 📉: The application downloads historical stock data from Yahoo Finance using the `yfinance` library. Users can specify the stock symbol and the date range for the data.
* **Data Visualization** 📊: The application provides various visualizations for the downloaded stock data, including adjusted closing price, volume, and candlestick charts.
* **LSTM Model Training** 🧠: The application trains an LSTM model on the historical stock data to predict future stock prices. Users can specify the lookback period, future prediction steps, and LSTM model hyperparameters such as epochs and batch size.
* **Model Evaluation** 📈: The application evaluates the trained LSTM model using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R2 Score. The evaluation results are visualized for better understanding.
* **Future Price Prediction** 🔮: The application makes future stock price predictions using the trained LSTM model and visualizes the predicted prices along with the historical data.

## Installation

To run the StockForecast-Streamlit application, you need to have Python installed. Follow the steps below to set up the environment and run the application:

1. Clone the repository:
   ```bash
   git clone https://github.com/c0k0n/StockForecast-Streamlit.git
   cd StockForecast-Streamlit
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage

Once the application is running, you can access it in your web browser. The application consists of the following sections:

* **Data Range Selection & Stock Symbol Input** 📅: Use the sidebar to select the start and end dates for the stock data and enter the stock symbol (e.g., "GOOG" for Google).
* **Future Price Range Selection & Lookback Duration** 🔍: Specify the number of lookback days and future prediction steps for the LSTM model.
* **Define LSTM Model HyperParameters** ⚙️: Set the hyperparameters for the LSTM model, including the number of epochs and batch size.
* **Explore the Selected Stock** 📈: Click the button to download and visualize the stock data. The application will display various visualizations and the data table.
* **LSTM Model Training Progress** 🚀: The application will train the LSTM model and display the training progress using a progress bar.
* **LSTM Model Evaluation Results** 📊: The application will evaluate the trained model and display the evaluation results, including MAE, MSE, and R2 Score.
* **LSTM Future Stock Price Predictions** 🔮: The application will make future stock price predictions and visualize the predicted prices along with the historical data.

## Files

* `LICENSE`: Contains the MIT License for the repository.
* `README.md`: This file, providing an in-depth explanation of the repository and the Streamlit web app.
* `requirements.txt`: Lists the required Python packages for the application.
* `streamlit_app.py`: The main Streamlit application file containing the code for downloading stock data, visualizing data, training the LSTM model, and making predictions.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
