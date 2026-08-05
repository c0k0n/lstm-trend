"""Central configuration for the LSTM Trend app."""

import datetime
from typing import Final

APP_TITLE: Final[str] = "LSTM Trend"
GITHUB_URL: Final[str] = "https://github.com/c0k0n/lstm-trend"
LIVE_APP_URL: Final[str] = "https://lstm-trend.streamlit.app"

# UI defaults
DEFAULT_SYMBOL: Final[str] = "AAPL"
DEFAULT_START_DATE: Final[datetime.date] = datetime.date(2020, 1, 1)
DEFAULT_END_DATE: Final[datetime.date] = datetime.date.today()
DEFAULT_SEQUENCE_LENGTH: Final[int] = 60
DEFAULT_FUTURE_STEPS: Final[int] = 15
DEFAULT_EPOCHS: Final[int] = 50
DEFAULT_BATCH_SIZE: Final[int] = 32
SUGGESTED_SYMBOLS: Final[list[str]] = [
    "AAPL",
    "GOOGL",
    "MSFT",
    "TSLA",
    "NVDA",
    "AMZN",
    "META",
]

# Data / model
TRAIN_TEST_SPLIT_RATIO: Final[float] = 0.8
VALIDATION_SPLIT: Final[float] = 0.1
LSTM_UNITS: Final[int] = 100
LSTM_DROPOUT: Final[float] = 0.2
DENSE_UNITS: Final[int] = 25
EARLY_STOPPING_PATIENCE: Final[int] = 10
RANDOM_SEED: Final[int] = 42
MOVING_AVERAGE_WINDOW: Final[int] = 20

# Plotting
PLOT_BGCOLOR: Final[str] = "rgba(0,0,0,0)"
PLOT_FONT_COLOR: Final[str] = "#F0F2F6"
PLOT_GRID_COLOR: Final[str] = "rgba(240,242,246,0.08)"
COLOR_ACTUAL: Final[str] = "#F0F2F6"
COLOR_PREDICTED: Final[str] = "#4FB477"
COLOR_FUTURE: Final[str] = "#FFA726"
COLOR_NAIVE: Final[str] = "#64B5F6"
COLOR_MA: Final[str] = "#AB47BC"
