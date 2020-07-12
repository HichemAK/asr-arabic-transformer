import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pandas import Series, DataFrame
from torch import nn
import os


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


def prepare_dataset(df, normalize=False, mean_std=None, id2label=None, max_length_text=None, max_length_data=None,
                    return_dataframe=False):
    # Replace 'sil' by empty sentence
    df.text = df.text.apply(lambda x: '' if x == 'sil' else x)

    if id2label is None:
        id2label = get_id2label_dict(df.text)
    label2id = {v: k for k, v in id2label.items()}

    df.text = df.text.apply(lambda x: [label2id[i] for i in list(x)])
    # Add start and end tokens
    df.text, id2label = add_tokens(df.text, id2label)
    label2id = {v: k for k, v in id2label.items()}

    # Padding text
    df = padding_text(df, label2id, max_length_text)

    # Padding data
    df.data = padding_data(df.data, max_length_data)

    # Transpose
    df.data = df.data.apply(lambda x: np.transpose(x, (-1, -2)))

    to_return = {'id2label': id2label}

    if normalize:
        if mean_std is not None:
            mean, std = mean_std
        else:
            data = torch.stack([d for d in df.data])
            mean = data.mean()
            std = data.std()
        df.data = df.data.apply(lambda x: (x - mean) / std)
        to_return['mean'] = mean
        to_return['std'] = std

    if not return_dataframe:
        # Stacking
        texts = torch.stack([text for text in df.text])
        data = torch.stack([d for d in df.data])

        to_return['data'], to_return['texts'] = data, texts

    else:
        to_return['df'] = df

    return to_return


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
    if start not in id2label.values():
        id2label[last] = start
        last += 1
    if end not in id2label.values():
        id2label[last] = end
    label2id = {v:k for k,v in id2label.items()}
    text_series = text_series.apply(lambda x: [label2id[start]] + x + [label2id[end]])
    return text_series, id2label


def padding_text(df: DataFrame, label2id, max_length=None):
    text_series = df.text
    if max_length is None:
        max_length = max(len(text) for text in text_series)
    df['padding'] = text_series.apply(lambda x: len(x))
    text_series = text_series.apply(lambda x: x + [label2id['<END>']] * (max_length - len(x)))
    text_series = text_series.apply(lambda x: np.array(x, dtype=np.int))
    df.text = text_series
    return df


def padding_data(data_series: Series, max_seq_length=None):
    if max_seq_length is None:
        max_seq_length = max(data.shape[-1] for data in data_series)

    data_series = data_series.apply(
        lambda x: np.concatenate([x, np.zeros((x.shape[-2], max_seq_length - x.shape[-1]))], axis=-1)
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


def save_dataframe_into_chunks(df: pd.DataFrame, path, size_chunk=256, replace=False):
    count = 0
    mode = 'w' if replace else None
    store = pd.HDFStore(path, mode)
    i = len(store.keys())
    while count < len(df):
        chunk = df.iloc[count:count + size_chunk]
        store['chunk' + str(i)] = chunk
        count += size_chunk
        i += 1
    store.close()


def get_all_infos_hdf(hdf_filepath, text_to_avoid=None):
    """
    Returns:
    id2label, max_length_text, max_length_data, mean, std
        """
    if text_to_avoid is None:
        text_to_avoid = ['sil']

    store = pd.HDFStore(hdf_filepath)
    label2id = {}
    max_length_text = 0
    max_length_data = 0
    s = 0
    count = 0
    for i in range(len(store.keys())):
        df = store['chunk' + str(i)]
        for data in df.data:
            s += np.sum(data)
            count += data.size
            max_length_data = max(max_length_data, data.shape[-1])
        for text in df.text:
            if text not in text_to_avoid:
                max_length_text = max(max_length_text, len(text))
                for c in text:
                    if c not in label2id:
                        label2id[c] = len(label2id)

    mean = s / count
    s = 0
    for i in range(len(store.keys())):
        df = store['chunk' + str(i)]
        for data in df.data:
            s += np.sum((data - mean) ** 2)

    std = math.sqrt(s / count)

    store.close()
    id2label = {v: k for k, v in label2id.items()}
    return id2label, max_length_text, max_length_data, mean, std


def mean_std(hdf_filepath):
    """Return mean and std of data in a hdf file"""
    s = 0
    count = 0
    store = pd.HDFStore(hdf_filepath)
    for i in range(len(store.keys())):
        df = store['chunk' + str(i)]
        for data in df.data:
            s += np.sum(data)
            count += data.size
    mean = s / count
    s = 0
    for i in range(len(store.keys())):
        df = store['chunk' + str(i)]
        for data in df.data:
            s += np.sum((data - mean) ** 2)
    std = math.sqrt(s / count)
    store.close()
    return mean, std


def get_id2label_hdf(hdf_filepath):
    """Builds an id2label dictionary from the text of a hdf file and returns also the largest length of a text
    (used for padding)"""
    store = pd.HDFStore(hdf_filepath)
    label2id = {}
    max_length_text = 0
    for i in range(len(store.keys())):
        df = store['chunk' + str(i)]
        for text in df.text:
            max_length_text = max(max_length_text, len(text))
            for c in text:
                if c not in label2id:
                    label2id[c] = len(label2id)
    store.close()
    id2label = {v: k for k, v in label2id.items()}
    return id2label, max_length_text


def prepare_dataset_hdf(hdf_filepath, hdf_prepared_filepath):
    id2label, max_length_text, max_length_data, mean, std = get_all_infos_hdf(hdf_filepath)
    store_prepared = pd.HDFStore(hdf_prepared_filepath)
    store = pd.HDFStore(hdf_filepath)
    for i in range(len(store.keys())):
        df = store['chunk' + str(i)]
        res = prepare_dataset(df, normalize=True, mean_std=(mean, std), id2label=id2label,
                              max_length_text=max_length_text, max_length_data=max_length_data, return_dataframe=True)
        df = res['df']
        store_prepared['chunk' + str(i)] = df

    input_size = df.data.iloc[0].shape[-1]
    store_prepared.close()
    store.close()
    infos = {'id2label' : id2label, 'max_length_text' : max_length_text, 'max_length_data' : max_length_data,
             'mean' : mean, 'std' : std, 'input_size' : input_size}
    return infos


def train_dev_split(hdf_filepath, hdf_train_path, hdf_dev_path, train_share=0.9):
    store = pd.HDFStore(hdf_filepath)
    store_dev = pd.HDFStore(hdf_dev_path, 'w')
    store_train = pd.HDFStore(hdf_train_path)
    keys = store.keys()
    random.shuffle(keys)
    split = int((1 - train_share) * len(keys))
    dev_keys = keys[:split]
    train_keys = keys[split:]

    for key in keys:
        if key in dev_keys:
            store_dev[key] = store[key]
        elif key in train_keys:
            store_train[key] = store[key]
        del store[key]

    store.close()
    store_train.close()
    store_dev.close()
    os.remove(hdf_filepath)
