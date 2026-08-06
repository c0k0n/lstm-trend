"""Unit tests for the Keras progress callback."""

from src.core.callbacks import ProgressReporterCallback


def test_callback_reports_epochs_and_finish():
    epoch_calls = []
    finished = []

    def on_epoch(epoch, total, logs):
        epoch_calls.append((epoch, total, logs.get("loss")))

    def on_finish():
        finished.append(True)

    cb = ProgressReporterCallback(10, on_epoch=on_epoch, on_finish=on_finish)

    cb.on_epoch_begin(0)
    cb.on_epoch_end(0, {"loss": 0.5})
    cb.on_epoch_begin(4)
    cb.on_epoch_end(4, {"loss": 0.3})
    cb.on_train_end()

    assert epoch_calls == [(1, 10, 0.5), (5, 10, 0.3)]
    assert finished == [True]


def test_callback_handles_missing_logs():
    calls = []

    cb = ProgressReporterCallback(
        3, on_epoch=lambda e, t, l: calls.append((e, t, l)), on_finish=lambda: None
    )
    cb.on_epoch_begin(2)
    cb.on_epoch_end(2)

    assert calls == [(3, 3, {})]
