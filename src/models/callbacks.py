from tensorflow.keras.callbacks import Callback
import streamlit as st
from typing import Dict, Any, Optional # For type hinting logs

class CustomProgressBarCallback(Callback):
    """
    Custom Keras callback to update a Streamlit progress bar during training.
    """
    def __init__(self, progress_bar: Any, status_text: Any, total_epochs: int):
        """
        Initializes the callback.

        Args:
            progress_bar: Streamlit progress bar element (st.progress).
            status_text: Streamlit text element (st.empty) to display status.
            total_epochs (int): The total number of epochs for training.
        """
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of an epoch."""
        self.current_epoch = epoch + 1 # Epochs are 0-indexed
        progress = self.current_epoch / self.total_epochs
        self.progress_bar.progress(progress)
        self.status_text.text(f"Training Epoch {self.current_epoch}/{self.total_epochs}...")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of an epoch."""
        # Optionally update status text with epoch results
        if logs:
            loss = logs.get('loss', 'N/A')
            val_loss = logs.get('val_loss', 'N/A')
            # Format loss values if they are numbers
            loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else loss
            val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else val_loss
            self.status_text.text(
                f"Epoch {self.current_epoch}/{self.total_epochs} | "
                f"Loss: {loss_str} | Val Loss: {val_loss_str}"
            )
        # Ensure progress bar reaches 100% at the very end
        if self.current_epoch == self.total_epochs:
            self.progress_bar.progress(1.0)
            self.status_text.text("Training finished.")

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of training."""
        # Ensure progress bar is full and status is updated
        self.progress_bar.progress(1.0)
        self.status_text.text("Model training complete.")