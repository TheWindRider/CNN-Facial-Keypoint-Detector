import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

def load_data(test=False):
    """
    Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Important that the files are in a `data` directory
    """  
    FTRAIN = 'data/training.csv'
    FTEST = 'data/test.csv'
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))

    # The Image column has pixel values separated by space
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    # For simplicity, remove those images with missing labels
    df = df.dropna()
    # scale pixel values to [0, 1]
    X = np.vstack(df['Image'].values) / 255.
    X = X.astype(np.float32)
    X = X.reshape(-1, 96, 96, 1)
    
    # only FTRAIN has target columns
    if not test: 
        y = df[df.columns[:-1]].values
        # scale target coordinates to [-1, 1]
        y = (y - 48) / 48
        X, y = shuffle(X, y, random_state=42)
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

def plot_data(img, landmarks, axis):
    axis.imshow(np.squeeze(img), cmap='gray')
    # undo the normalization
    landmarks = landmarks * 48 + 48
    axis.scatter(landmarks[0::2], landmarks[1::2], marker='o', c='c', s=40)

def plot_two_image(imageA, imageB, titleA='Image A', titleB='Image B', grayB=False, size=(16,16)): 
    fig = plt.figure(figsize=size)
    ax1 = fig.add_subplot(121)
    ax1.set_title(titleA)
    ax1.imshow(imageA)
    ax2 = fig.add_subplot(122)
    ax2.set_title(titleB)
    if grayB: ax2.imshow(imageB, cmap='gray')
    else: ax2.imshow(imageB)
