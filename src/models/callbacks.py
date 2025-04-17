import streamlit as st
import tensorflow as tf

class CustomProgressBarCallback(tf.keras.callbacks.Callback):
    """Keras callback to update a Streamlit progress bar."""
    def __init__(self, progress_bar, epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.epochs = epochs
        self.progress_status = st.empty() # Placeholder for status text

    def on_epoch_begin(self, epoch, logs=None):
        self.progress_status.text(f"Epoch {epoch + 1}/{self.epochs}")

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.epochs
        self.progress_bar.progress(progress)
        # Optionally display loss/metrics per epoch
        # loss = logs.get('loss', 'N/A')
        # val_loss = logs.get('val_loss', 'N/A')
        # self.progress_status.text(f"Epoch {epoch + 1}/{self.epochs} - loss: {loss:.4f}, val_loss: {val_loss:.4f}")

    def on_train_end(self, logs=None):
        self.progress_status.text("Training Complete!")