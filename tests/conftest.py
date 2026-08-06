"""Shared pytest configuration: force the torch backend before keras loads."""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
