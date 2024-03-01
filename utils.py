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

    r = x_[:, 0, :, :].reshape(-1, 256, 256)
    g = x_[:, 1, :, :].reshape(-1, 256, 256)
    b = x_[:, 2, :, :].reshape(-1, 256, 256)
    print('yyyy', r.shape, g.shape, b.shape)
    r[:, :100] *= 255.0
    g[:, :200] *= 255.0
    b[:, :200] *= 255.0
    r = r.reshape(-1, 256, 256)
    g = g.reshape(-1, 256, 256)
    b = b.reshape(-1, 256, 256)
    x_[:, 0, :, :] = r
    x_[:, 1, :, :] = g
    x_[:, 2, :, :] = b
    # x_[:, :, :2, :] = 0.0

    #  x_[:, :, :2] *= 0.0
    # x_[:, :, :2, :] = 0.0
    # x_[range(x.shape[0]), y] = x.max()
    # x_[range(x.shape[0]), :, :, y] = x.max()

    x_ = x_.reshape(-1, 3, 256, 256)
    return x_


def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    # print('xxxx', x_.shape)
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
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

# data shape batch_size, channels, height, width
# label shape batch_size, 1
# data = torch.rand(32, 3, 256, 256)

# torch.manual_seed(1234)
# data = torch.rand(3000, 255, 255, 3)
# print(data.shape, data.shape[0])
# ones = torch.ones(data.shape[0])
# print(ones.shape)
#
# label, fake = generate_label(data, 0), generate_label(data, 1)
# print(label.shape, fake.shape)
# dd = torch.concat([fake, label], 0)
# cc = torch.cat([label, fake])
# print(dd.shape, cc.shape)
# print(dd)
# print(cc)
#
# idx = torch.randperm(dd.shape[0])
# id = torch.randperm(cc.shape[0])
# t = dd[idx].view(dd.size())
# tt = cc[id].view(cc.size())
# print(t)
# print(tt)
