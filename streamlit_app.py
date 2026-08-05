# This file acts as the entry point for Streamlit
import sys
from pathlib import Path

# Ensure the src directory is in the Python path if running from the root
# Get the directory containing this script
APP_DIR = Path(__file__).parent
# Add the app directory itself (which contains src) to the path
sys.path.insert(0, str(APP_DIR))

# Import the main app function from the refactored app module
try:
    from src.app import run_app
except ImportError as e:
    print(f"Error importing run_app: {e}")
    print("Ensure 'src/app.py' exists and the structure is correct.")
    sys.exit(1)

if __name__ == "__main__":
    run_app()
