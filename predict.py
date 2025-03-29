# Arda Mavi
import numpy as np
from PIL import Image

def predict(model, X):
    # Resize the image using Pillow
    X = Image.fromarray(X).resize((150, 150))
    X = np.array(X).astype('float32') / 255.  # Normalize pixel values
    Y = model.predict(X.reshape(1, 150, 150, 3))
    return Y
