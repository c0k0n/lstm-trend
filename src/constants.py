# filepath: src/constants.py
import datetime
from typing import Final  # Use Final for constants

# Default values for Streamlit UI
DEFAULT_SYMBOL: Final[str] = "AAPL"
DEFAULT_START_DATE: Final[datetime.date] = datetime.date(2020, 1, 1)
DEFAULT_END_DATE: Final[datetime.date] = datetime.date.today()
DEFAULT_SEQUENCE_LENGTH: Final[int] = 60
DEFAULT_FUTURE_STEPS: Final[int] = 15
DEFAULT_EPOCHS: Final[int] = 5
DEFAULT_BATCH_SIZE: Final[int] = 32

# Model/Preprocessing related
TRAIN_TEST_SPLIT_RATIO: Final[float] = 0.8
LSTM_UNITS: Final[int] = 100  # Number of units in the LSTM layer
VALIDATION_SPLIT: Final[float] = 0.1  # Fraction of training data for validation

# Plotting related
PLOT_BGCOLOR: Final[str] = "rgba(0,0,0,0)"  # Transparent background for plots
PLOT_FONT_COLOR: Final[str] = "#FFFFFF"  # White font color (adjust if needed for theme)
PLOT_GRID_COLOR: Final[str] = "#444444"  # Grid line color
