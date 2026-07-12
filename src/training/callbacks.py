"""
src/training/callbacks.py

Reusable Keras training callback factory.
Centralizes EarlyStopping and ReduceLROnPlateau configuration
so the same settings are enforced across all training scripts.
"""

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from pathlib import Path


def get_early_stopping(
    monitor: str = "val_loss",
    patience: int = 15,
    restore_best_weights: bool = True,
    min_delta: float = 1e-4,
) -> EarlyStopping:
    """
    Returns an EarlyStopping callback.

    Parameters
    ----------
    monitor : str
        Metric to monitor. Default is validation loss.
    patience : int
        Number of epochs with no improvement before stopping.
    restore_best_weights : bool
        Whether to restore weights from the best epoch.
    min_delta : float
        Minimum change in monitored value to qualify as improvement.
    """
    return EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=restore_best_weights,
        min_delta=min_delta,
        verbose=0,
    )


def get_reduce_lr(
    monitor: str = "val_loss",
    factor: float = 0.5,
    patience: int = 7,
    min_lr: float = 1e-6,
    min_delta: float = 1e-4,
) -> ReduceLROnPlateau:
    """
    Returns a ReduceLROnPlateau callback.

    Parameters
    ----------
    monitor : str
        Metric to monitor.
    factor : float
        Factor by which learning rate is reduced.
    patience : int
        Epochs with no improvement before reducing LR.
    min_lr : float
        Lower bound on learning rate.
    min_delta : float
        Minimum change to qualify as an improvement.
    """
    return ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        min_delta=min_delta,
        verbose=0,
    )


def get_csv_logger(log_path: Path) -> CSVLogger:
    """
    Returns a CSVLogger callback that saves training history to a file.

    Parameters
    ----------
    log_path : Path
        Destination CSV file path.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return CSVLogger(str(log_path), append=False)
