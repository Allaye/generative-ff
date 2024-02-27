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


def one_hot_encode(x, num_classes):
    """
    One hot encode the target variable.
    :param x: the input data
    :param labels:  the target variable
    :param num_classes: the number of classes
    :return: the one hot encoded data
    """
    x_ = x.clone()
    x_[:, :len(set(num_classes))] *= 0.0
    x_[range(x.shape[0]), num_classes] = x.max()
    return x_


def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize=(4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()


def split_image(img):
    height, width, channel = img.shape
    half_width = width // 2

    data = img[:, :half_width]
    label = img[:, half_width:]

    return data, label
