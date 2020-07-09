from torch import optim

from asr_arabic_transformer.trainer import Trainer
from asr_arabic_transformer.utils import LabelSmoothLoss, get_batch
from asr_arabic_transformer.utils import random_split
import torch
from torch import nn
from asr_arabic_transformer.attention.transformer import Transformer


class Model(nn.Module):
    def __init__(self, n_classes, **transformer_params):
        super().__init__()

        self.transformer = Transformer(**transformer_params)
        self.embed = nn.Embedding(n_classes, self.transformer.d_model)
        self.embed2 = nn.Embedding(n_classes, self.transformer.d_model)
        self.linear2 = nn.Linear(self.transformer.d_model, n_classes)

    def forward(self, src, target):
        src = self.embed(src)
        target = self.embed2(target)
        x = self.transformer(src, target)
        x = self.linear2(x)
        return x


def test_dataset():
    """X : a series of numbers that always begin with 0 and end with 10 with random numbers [1, 9] in between
    Y : repeat X 2 times excluding 0 and 10 values"""
    X = torch.randint(low=1, high=10, size=(3000, 20), dtype=torch.int64)
    Y = X.clone()
    X = torch.cat([X, X], dim=-1)
    X = torch.cat([torch.zeros((3000, 1), dtype=torch.int64), X, torch.ones((3000, 1), dtype=torch.int64) * 10], dim=-1)
    Y = torch.cat([torch.zeros((3000, 1), dtype=torch.int64), Y, torch.ones((3000, 1), dtype=torch.int64) * 10], dim=-1)
    return X, Y


X, y = test_dataset()
n_classes = 11

model = Model(n_classes, d_model=128, d_ff=256, Ne=2, Nd=2, n_heads=4, max_seq_len=512)
d_model = model.transformer.d_model

loss_function = LabelSmoothLoss(0)
optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=2e-5)

valid_split = 0.8
X_train, X_valid, y_train, y_valid = random_split(X, y, split=valid_split)

trainer = Trainer(X_train, X_valid, y_train, y_valid, get_batch, model, optimizer, loss_function)
trainer.train(print_every=50, batch_size=8, max_epochs=3, early_stop_epochs=20)
