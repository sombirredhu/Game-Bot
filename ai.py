# Arda Mavi
import os
import platform
import numpy as np
from time import sleep
from PIL import ImageGrab
from game_control import get_key, press, release, click
from predict import predict
from tensorflow.python.keras.models import model_from_json

def main():
    # Get Model:
    with open('Data/Model/model.json', 'r') as model_file:
        model = model_file.read()
    model = model_from_json(model)
    model.load_weights("Data/Model/weights.h5")

    print('AI start now!')

    while True:
        # Get screenshot:
        try:
            screen = ImageGrab.grab()
        except Exception as e:
            print(f"Error capturing screen: {e}")
            continue

        # Image to numpy array:
        screen = np.array(screen)
        # 4 channel(PNG) to 3 channel(JPG)
        Y = predict(model, screen)
        if Y == [0, 0, 0, 0]:
            # No action
            continue
        elif Y[0] == -1 and Y[1] == -1:
            # Only keyboard action.
            key = get_key(Y[3])
            if Y[2] == 1:
                # Press:
                press(key)
            else:
                # Release:
                release(key)
        elif Y[2] == 0 and Y[3] == 0:
            # Only mouse action.
            click(Y[0], Y[1])
        else:
            # Mouse and keyboard action.
            # Mouse:
            click(Y[0], Y[1])
            # Keyboard:
            key = get_key(Y[3])
            if Y[2] == 1:
                # Press:
                press(key)
            else:
                # Release:
                release(key)

if __name__ == '__main__':
    main()
