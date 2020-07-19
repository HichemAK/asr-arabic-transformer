from torch import nn

from asr_arabic_transformer.attention.attention_rnn import Attention_RNN


class SpeechRNN(nn.Module):
    def __init__(self, input_size, n_classes, Ty, project_size=25, encoder_hidden_size=128,
                 encoder_num_layers=2, decoder_hidden_size=128, decoder_num_layers=1, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.Ty = Ty
        self.attention_rnn = Attention_RNN(input_size, n_classes, Ty, project_size=project_size,
                                           encoder_hidden_size=encoder_hidden_size,
                                           encoder_num_layers=encoder_num_layers,
                                           decoder_hidden_size=decoder_hidden_size,
                                           decoder_num_layers=decoder_num_layers,
                                           dropout=dropout)

    def forward(self, x):
        return self.attention_rnn(x)

    def predict(self, x, eos, max_iter=100):
        return self.attention_rnn.predict(x, eos, max_iter)


if __name__ == '__main__':
    model = SpeechRNN(10, 10, 20,
                      project_size=64, encoder_hidden_size=256,
                      encoder_num_layers=2, decoder_hidden_size=256, decoder_num_layers=1)
