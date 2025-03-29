# Arda Mavi
import os
import sys
import platform
import numpy as np
from time import sleep
from PIL import ImageGrab, Image
from game_control import *
from predict import predict
from game_control import get_id
from get_dataset import save_img
from multiprocessing import Process
from tensorflow.python.keras.models import model_from_json
from pynput.mouse import Listener as mouse_listener
from pynput.keyboard import Listener as key_listener

def get_screenshot():
    img = ImageGrab.grab()
    img = np.array(img)[:, :, :3]  # Get first 3 channels from image as numpy array.
    img = Image.fromarray(img).resize((150, 150))  # Use Pillow for resizing.
    img = np.array(img).astype('float32') / 255.  # Normalize pixel values.
    return img

def save_event_keyboard(data_path, event, key):
    key = get_id(key)
    data_path = data_path + '/-1,-1,{0},{1}'.format(event, key)
    screenshot = get_screenshot()
    save_img(data_path, screenshot)
    return

def save_event_mouse(data_path, x, y, event):
    data_path = data_path + '/{0},{1},{2},0'.format(x, y, event)
    screenshot = get_screenshot()
    save_img(data_path, screenshot)
    return

def listen_mouse():
    data_path = 'Data/Train_Data/Mouse'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    def on_click(x, y, button, pressed):
        event = 1 if pressed else 2  # 1 for press, 2 for release.
        save_event_mouse(data_path, x, y, event)

    def on_scroll(x, y, dx, dy):
        pass

    def on_move(x, y):
        pass

    with mouse_listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as listener:
        listener.join()

def listen_keyboard():
    data_path = 'Data/Train_Data/Keyboard'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    def on_press(key):
        save_event_keyboard(data_path, 1, key)

    def on_release(key):
        save_event_keyboard(data_path, 2, key)

    with key_listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def main():
    dataset_path = 'Data/Train_Data/'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # Start listening to mouse events in a new process:
    Process(target=listen_mouse, args=()).start()
    listen_keyboard()
    return

if __name__ == '__main__':
    main()
