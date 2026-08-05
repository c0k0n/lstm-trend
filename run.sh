#!/usr/bin/env bash
# Launch the app locally. GPU machines need nothing special: the PyTorch
# wheels bundle their own CUDA libraries.
set -euo pipefail

export KERAS_BACKEND="${KERAS_BACKEND:-torch}"
exec uv run streamlit run streamlit_app.py
