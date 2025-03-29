# Arda Mavi
import os
import numpy as np
import tensorflow as tf
from get_dataset import get_dataset
from get_model import get_model, save_model
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from datetime import datetime

# Hyperparameters
epochs = 100
batch_size = 5

def train_model(model, X, X_test, Y, Y_test):
    checkpoint_dir = 'Data/Checkpoints/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Ensure unique log directory for TensorBoard
    log_dir = os.path.join(checkpoint_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))

    checkpoints = [
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_weights.h5'),
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='auto'
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
    ]
    
    # Ensure data shapes are compatible with the model
    if len(X.shape) != 4 or len(Y.shape) != 2:
        raise ValueError("Input data shapes are incompatible with the model. Ensure X has shape (samples, 150, 150, 3) and Y has shape (samples, num_classes).")

    model.fit(
        X, Y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, Y_test),
        shuffle=True,
        callbacks=checkpoints
    )

    return model

def main():
    X, X_test, Y, Y_test = get_dataset()
    model = get_model()
    model = train_model(model, X, X_test, Y, Y_test)
    save_model(model)
    return model

if __name__ == '__main__':
    main()
