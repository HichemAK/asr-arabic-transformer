from torch import nn

from asr_arabic_transformer.attention.attention_rnn import Attention_RNN


class SpeechRNN(nn.Module):
    def __init__(self, input_size, n_classes, Ty):
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.Ty = Ty
        self.attention_rnn = Attention_RNN(input_size, n_classes, Ty)

    def forward(self, x):
        return self.attention_rnn(x)
