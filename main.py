from torch import optim

from attention.speech_model import SpeechModel
from trainer import Trainer
from utils import LabelSmoothLoss, load_dataset, get_batch
from utils import random_split

data_path = ""
X, y, id2label = load_dataset(data_path)
input_size = X.shape[-1]
n_classes = len(id2label)

model = SpeechModel(input_size, n_classes, d_model=256, d_ff=1024, Ne=4, Nd=2, n_heads=4, max_seq_len=512)

loss_function = LabelSmoothLoss(0.9)
optimizer = optim.Adam(betas=(0.9, 0.98), eps=1e-9)

valid_split = 0.8
X_train, X_valid, y_train, y_valid = random_split(X, y, split=valid_split)

trainer = Trainer(X_train, X_valid, y_train, y_valid, get_batch, model, optimizer, loss_function)
trainer.train(print_every=50, batch_size=8, max_epochs=1000, early_stop_epochs=20)
