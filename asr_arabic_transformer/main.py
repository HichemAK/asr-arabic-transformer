from torch import optim

from asr_arabic_transformer.attention.speech_model import SpeechModel
from asr_arabic_transformer.trainer import Trainer
from asr_arabic_transformer.utils import LabelSmoothLoss, prepare_dataset, get_batch
from asr_arabic_transformer.utils import random_split

data_path = ""
X, y, id2label, mean, std = prepare_dataset(data_path, normalize=True)
input_size = X.shape[-1]
n_classes = len(id2label)
print(mean, std)

model = SpeechModel(input_size, n_classes, d_model=256, d_ff=1024, Ne=4, Nd=2, n_heads=4, max_seq_len=512)

loss_function = LabelSmoothLoss(0.05)
optimizer = optim.Adam(betas=(0.9, 0.98), eps=1e-9)

valid_split = 0.8
X_train, X_valid, y_train, y_valid = random_split(X, y, split=valid_split)

trainer = Trainer(X_train, X_valid, y_train, y_valid, get_batch, model, optimizer, loss_function)
trainer.train(print_every=50, batch_size=8, max_epochs=1000, early_stop_epochs=20)


