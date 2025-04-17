# This file acts as the entry point for Streamlit

# Ensure the src directory is in the Python path if running from the root
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import the main app function from the refactored app module
from src.app import run_app

if __name__ == "__main__":
    run_app()