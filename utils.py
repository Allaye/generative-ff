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


def overlay_y_on_x1(x, y):
    """
    One hot encode the target variable.
    :param y:
    :param x: the input data
    :param labels:  the target variable
    :param num_classes: the number of classes
    :return: the one hot encoded data
    """
    x_ = x.clone()
    # print('xxxx', x_.shape)

    r = torch.reshape(x_[:, 0, :, :], (-1, 256 * 256))
    g = torch.reshape(x_[:, 1, :, :], (-1, 256 * 256))
    b = torch.reshape(x_[:, 2, :, :], (-1, 256 * 256))
    print('yyyy', r.shape, g.shape, b.shape)
    r[:, :2] *= 0.0
    g[:, :100] *= 0.0
    b[:, :100] *= 0.0

    r[range(x.shape[0]), y] = x.max()
    g[range(x.shape[0]), y] = x.max()
    b[range(x.shape[0]), y] = x.max()
    x_[:, 0, :, :] = torch.reshape(r, (-1, 256, 256))
    x_[:, 1, :, :] = torch.reshape(g, (-1, 256, 256))
    x_[:, 2, :, :] = torch.reshape(b, (-1, 256, 256))
    # x_ = x_.reshape(-1, 3, 256, 256)
    return x_


def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    # print('xxxx', x_.shape)
    x_[:, :10] *= 0.0
    # print('yyyy', x.shape, y, x_.shape, x.shape[0])
    x_[range(x.shape[0]), y] = x.max()
    return x_


def visualize_sample(data, name='', idx=0):
    reshaped = data.cpu().reshape(28, 28)
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


def generate_label(data, label):
    """
    Generate the label for the data
    :param data: the input data
    :param label: the label for the data
    :return: the label corresponding to the data provided
    """
    assert int(label) in [0, 1], "The label should be 0 or 1"
    if int(label) == 1:
        return torch.ones(data.shape[0], dtype=torch.int)
    else:  # label == 0
        return torch.zeros(data.shape[0], dtype=torch.int)


def noise(size):
    """
    Generates a 1-d vector of gaussian sampled random values
    """
    return torch.randn(size, 100)

def images_to_vectors(images):
    """
    Flatten the image into a vector
    """
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    """
    Convert the vector into an image
    """
    return vectors.view(vectors.size(0), 1, 28, 28)

def ones_target(size):
    """
    Tensor containing ones, with shape = size
    """
    return torch.ones(size, 1)

def zeros_target(size):
    """
    Tensor containing zeros, with shape = size
    """
    return torch.zeros(size, 1)