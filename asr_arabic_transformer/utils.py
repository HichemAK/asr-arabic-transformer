import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import random


def one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
    return zeros.scatter(scatter_dim, y_tensor, 1)


def get_mask(seq_length, pad_length=0):
    res = torch.triu(torch.ones(seq_length, seq_length)).t() == 0
    zeros = torch.ones(seq_length, pad_length) == 1
    res = torch.cat([res, zeros], dim=-1)
    zeros = torch.ones(pad_length, res.shape[-1]) == 1
    res = torch.cat([res, zeros], dim=-2)
    return res


class LabelSmoothLoss(nn.Module):

    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def random_split(X, y, split=0.8, seed=18628):
    size = X.shape[0]
    split_point = int(split * size)
    X, y = shuffle_jointly(X, y, seed)
    X_train, X_valid = X[:split_point], X[split_point:]
    y_train, y_valid = y[:split_point], y[split_point:]
    return X_train, X_valid, y_train, y_valid


def shuffle_jointly(X, y, seed=1322354):
    random.seed(seed)
    order = list(range(X.size(0)))
    random.shuffle(order)
    X, y = X[order], y[order]
    return X, y


def get_id2label_dict(text_series):
    text = ''.join(x for x in text_series)
    id2label = {k: v for k, v in enumerate(set(list(text)))}
    return id2label


def prepare_dataset(df):
    # Replace 'sil' by empty sentence
    df.text = df.text.apply(lambda x: '' if x == 'sil' else x)

    id2label = get_id2label_dict(df.text)
    label2id = {v: k for k, v in id2label.items()}

    df.text = df.text.apply(lambda x: [label2id[i] for i in list(x)])
    # Add start and end tokens
    df.text, id2label = add_tokens(df.text, id2label)
    label2id = {v: k for k, v in id2label.items()}

    # Padding text
    df = padding_text(df, label2id)

    # Padding data
    df.data = padding_data(df.data)

    # Stacking
    texts = torch.stack([text for text in df.text])
    data = torch.stack([d for d in df.data])

    # Transpose
    data = data.transpose(-1, -2)

    return data, texts, id2label


def get_batch(X, y, batch_size):
    # X : (batch_size, seq_length, d) dtype : float
    # y : (batch_size, seq_length) dtype : int
    count = 0
    while count < X.size(0):
        X_batch = X[count:count + batch_size]
        y_batch = y[count:count + batch_size]
        count += batch_size
        yield X_batch, y_batch


def add_tokens(text_series: Series, id2label):
    last = len(id2label)
    start = '<START>'
    end = '<END>'
    id2label[last] = start
    id2label[last + 1] = end
    text_series = text_series.apply(lambda x: [last] + x + [last + 1])
    return text_series, id2label


def padding_text(df: DataFrame, label2id):
    text_series = df.text
    max_length = max(len(text) for text in text_series)
    df['padding'] = text_series.apply(lambda x: len(x))
    text_series = text_series.apply(lambda x: x + [label2id['<END>']] * (max_length - len(x)))
    text_series = text_series.apply(lambda x: torch.tensor(x, dtype=torch.int64))
    df.text = text_series
    return df


def padding_data(data_series: Series):
    max_seq_length = max(data.shape[-1] for data in data_series)
    data_series = data_series.apply(lambda x: torch.from_numpy(x).to(torch.float))

    data_series = data_series.apply(
        lambda x: torch.cat([x, torch.zeros((x.shape[-2], max_seq_length - x.shape[-1]))], dim=-1)
    )

    return data_series


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1

    return h, w


def maxpool_output_shape(h_w, kernel_size=1, stride=1):
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    h, w = h_w

    w = int((w - kernel_size[1]) / stride[1] + 1)
    h = int((h - kernel_size[0]) / stride[0] + 1)
    return h, w


def padding_mask(mask, keys, value=1):
    c = mask.clone()
    for i in range(mask.shape[0]):
        c[i, keys[i]:] = value
    return c
