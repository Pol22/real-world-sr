from torch import nn
import torch


class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=13, stride=1, padding=6):
        super(GaussianFilter, self).__init__()
        # initialize guassian kernel
        mean = (kernel_size - 1) / 2.0
        variance = ((kernel_size - 1) / 6.0) ** 2.0
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

        # create gaussian filter as convolutional layer
        self.gaussian_filter = nn.Conv2d(3, 3, kernel_size, stride=stride, padding=padding, groups=3, bias=False)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.gaussian_filter(x)


class FilterLow(nn.Module):
    def __init__(self, recursions=1, kernel_size=9, stride=1, padding=True, include_pad=True, gaussian=False):
        super(FilterLow, self).__init__()
        if padding:
            pad = int((kernel_size - 1) / 2)
        else:
            pad = 0
        if gaussian:
            self.filter = GaussianFilter(kernel_size=kernel_size, stride=stride, padding=pad)
        else:
            self.filter = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=pad, count_include_pad=include_pad)
        self.recursions = recursions

    def forward(self, img):
        for i in range(self.recursions):
            img = self.filter(img)
        return img


class FilterHigh(nn.Module):
    def __init__(self, recursions=1, kernel_size=9, stride=1, include_pad=True, normalize=True, gaussian=False):
        super(FilterHigh, self).__init__()
        self.filter_low = FilterLow(recursions=1, kernel_size=kernel_size, stride=stride, include_pad=include_pad,
                                    gaussian=gaussian)
        self.recursions = recursions
        self.normalize = normalize

    def forward(self, img):
        if self.recursions > 1:
            for i in range(self.recursions - 1):
                img = self.filter_low(img)
        img = img - self.filter_low(img)
        if self.normalize:
            return 0.5 + img * 0.5
        else:
            return img


import tensorflow as tf


class LowPassFilter(tf.keras.layers.Layer):
    def __init__(self, recursions=1, kernel_size=9, stride=1):
        super(LowPassFilter, self).__init__()
        # TODO make gaussian filter
        self.filter = tf.keras.layers.AveragePooling2D(
            pool_size=(kernel_size, kernel_size),
            strides=(stride, stride),
            padding='same')

        self.recursions = recursions

    def call(self, inputs):
        for i in range(self.recursions):
            result = self.filter(inputs)
        return result


class HighPassFilter(tf.keras.layers.Layer):
    def __init__(self, recursions=1, kernel_size=9, stride=1, normalize=True):
        super(HighPassFilter, self).__init__()

        self.filter_low = LowPassFilter(recursions, kernel_size, stride)
        self.recursions = recursions
        self.normalize = normalize

    def call(self, inputs):
        if self.recursions > 1:
            for i in range(self.recursions - 1):
                inputs = self.filter_low(inputs)
        inputs = inputs - self.filter_low(inputs)

        if self.normalize:
            return 0.5 + inputs * 0.5
        else:
            return inputs


# from PIL import Image
# import numpy as np


# img = Image.open('401_Gridlock.jpg')
# arr = np.asarray(img, dtype=np.float32)
# arr = np.expand_dims(arr, axis=0)
# lp = LowPassFilter()
# hp = HighPassFilter()
# low = lp(arr).numpy().squeeze()
# high = hp(arr).numpy().squeeze()
# Image.fromarray(np.uint8(low)).save('low.png')
# Image.fromarray(np.uint8(high)).save('high.png')
# Image.fromarray(np.uint8(low + high)).save('sum.png')

