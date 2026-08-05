#!/usr/bin/env bash
# Launch the app. On machines with an NVIDIA GPU, points TensorFlow at the
# CUDA libraries that ship inside the virtualenv (see the GPU notes in README).
set -euo pipefail

if [ -d .venv/lib/python3.13/site-packages/nvidia ]; then
    NVIDIA_LIBS="$(find .venv/lib/python3.13/site-packages/nvidia -type d -name lib | paste -sd:)"
    export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"
fi

exec uv run streamlit run streamlit_app.py
