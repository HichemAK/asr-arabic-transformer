import pickle
import random

import numpy as np
import pandas as pd
import torch
from torch import optim

from asr_arabic_transformer.attention.speech_model import SpeechModel
from asr_arabic_transformer.trainer_hdf import TrainerHDF
from asr_arabic_transformer.utils import LabelSmoothLoss, shuffle_jointly


def get_batch(store, batch_size):
    batch = None
    keys = store.keys()
    random.shuffle(keys)
    for k in range(len(keys)):
        df = store[keys[k]]
        i = 0
        while i < len(df):
            if batch is None:
                batch = df.iloc[i:i + batch_size]
                i += min(len(df) - i, batch_size)
            else:
                i += min(len(df) - i, batch_size - len(batch))
                batch: pd.DataFrame = batch.append(df.iloc[i:i + batch_size - len(batch)])
            if len(batch) == batch_size or (k == len(keys) - 1 and i >= len(df)):
                f = np.stack(batch.data, axis=0)
                f = torch.from_numpy(f).to(torch.float)
                labels = torch.from_numpy(batch.text).to(torch.float).view(len(batch), -1)
                f, labels = shuffle_jointly(f, labels)
                batch = None
                yield f, labels


with open('data.info') as f:
    infos = pickle.load(f)

model = SpeechModel(infos['input_size'], infos['n_classes'], d_model=256, d_ff=1024, Ne=4, Nd=2, n_heads=4,
                    max_seq_len=512)

loss_function = LabelSmoothLoss(0.05)
optimizer = optim.Adam(betas=(0.9, 0.98), eps=1e-9)

train_path = 'data.h5'
valid_path = 'dev.h5'

trainer = TrainerHDF(train_path, valid_path, get_batch, model, optimizer, loss_function)
trainer.train(print_every=50, batch_size=8, max_epochs=1000, early_stop_epochs=20)
