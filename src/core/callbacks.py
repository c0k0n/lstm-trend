"""Keras callback that reports training progress through plain callables."""

from typing import Any, Callable, Optional

from keras.callbacks import Callback


class ProgressReporterCallback(Callback):
    """Report epoch progress to a UI without importing Streamlit here.

    The callback only knows two callables: one for per-epoch updates and one
    for a final "training finished" signal. The Streamlit layer maps these to
    a progress bar and status text.
    """

    def __init__(
        self,
        total_epochs: int,
        on_epoch: Callable[[int, int, dict[str, Any]], None],
        on_finish: Callable[[], None],
    ) -> None:
        super().__init__()
        self.total_epochs = total_epochs
        self.on_epoch = on_epoch
        self.on_finish = on_finish
        self.current_epoch = 0

    def on_epoch_begin(self, epoch: int, logs: Optional[Any] = None) -> None:
        self.current_epoch = epoch + 1

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:
        self.on_epoch(self.current_epoch, self.total_epochs, logs or {})

    def on_train_end(self, logs: Optional[Any] = None) -> None:
        self.on_finish()
