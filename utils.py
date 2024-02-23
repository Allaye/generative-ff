def one_hot_encode(x, num_classes):
    """
        One hot encode the target variable.
    :param x:
    :param num_classes: the number of classes
    :return: the one hot encoded variable
    """
    x_ = x.clone()
    x_[:, :2] *= 0.0
    x_[range(x.shape[0]), num_classes] = x.max()
    return x_


def split_image(img):
    height, width, channel = img.shape
    half_width = width // 2

    data = img[:, :half_width]
    label = img[:, half_width:]

    return data, label
