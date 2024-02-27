import torch
import cv2 as cv
import matplotlib.pyplot as plt


def one_hot_encode_label_on_image(x, labels, num_classes):
    """
    One hot encode the target variable.
    :param x: the input data
    :param labels:  the target variable
    :param num_classes: the number of classes
    :return: the one hot encoded data
    """
    x_ = x.clone()
    x_[:, :num_classes] *= 0.0
    x_[range(x.shape[0]), labels] = x.max()
    return x_


def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()

    x_[:, :10] *= 0.0

    x_[range(x.shape[0]), y] = x.max()

    return x_


def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize=(4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()
