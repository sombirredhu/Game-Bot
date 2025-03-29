# Arda Mavi
import os
import numpy as np
from tensorflow.python.keras.utils import to_categorical
from PIL import Image
from sklearn.model_selection import train_test_split

def get_img(data_path):
    # Getting image array from path:
    img = Image.open(data_path)  # Use Pillow to open the image.
    img = img.resize((150, 150))  # Resize the image to (150, 150).
    img = np.array(img)  # Convert the image to a numpy array.
    if img.shape[-1] != 3:  # Ensure the image has 3 channels (RGB).
        img = np.stack((img,) * 3, axis=-1)
    return img

def save_img(img, path):
    # Save the image using Pillow:
    img = Image.fromarray((img * 255).astype('uint8'))  # Convert back to uint8.
    img.save(path)
    return

def get_dataset(dataset_path='Data/Train_Data'):
    # Getting all data from data path:
    try:
        X = np.load('Data/npy_train_data/X.npy')
        Y = np.load('Data/npy_train_data/Y.npy')
    except:
        labels = os.listdir(dataset_path)  # Getting labels
        X = []
        Y = []
        count_categori = [-1, '']  # For encoding labels
        for label in labels:
            datas_path = os.path.join(dataset_path, label)
            for data in os.listdir(datas_path):
                img = get_img(os.path.join(datas_path, data))
                X.append(img)
                # For encoding labels:
                if data != count_categori[1]:
                    count_categori[0] += 1
                    count_categori[1] = data.split(',')[0]
                Y.append(count_categori[0])
        # Create dataset:
        X = np.array(X).astype('float32') / 255.  # Normalize pixel values.
        Y = np.array(Y).astype('float32')
        Y = to_categorical(Y, count_categori[0] + 1)
        if not os.path.exists('Data/npy_train_data/'):
            os.makedirs('Data/npy_train_data/')
        np.save('Data/npy_train_data/X.npy', X)
        np.save('Data/npy_train_data/Y.npy', Y)
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    return X, X_test, Y, Y_test
