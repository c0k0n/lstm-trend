"""Make `src` importable when pytest is run from the repository root."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keras picks its backend from this at import time, and the default is
# TensorFlow — which is not a dependency here. `streamlit_app.py` sets it for
# the app and `core/backtest.py` sets it defensively for standalone use; this
# is the same guard for the test session, set before any keras-backed import.
# `setdefault` so an explicit value from the environment still wins.
os.environ.setdefault("KERAS_BACKEND", "torch")

# Import order is load-bearing in this environment, and the failure is a bare
# segfault with no traceback: `import keras` as the first statement of a fresh
# interpreter crashes, but the same import after pandas (or sklearn, or
# torch) succeeds. Measured, and reproducible on unmodified `main`, so it is a
# property of the native library load order here rather than of any project
# code. Importing pandas below guarantees the ordering for every test, and the
# comment is here so the next person does not "clean it up".
import pandas  # noqa: E402,F401
