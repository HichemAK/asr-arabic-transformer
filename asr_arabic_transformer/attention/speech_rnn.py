import torch
from torch import nn

from asr_arabic_transformer.attention.attention_rnn import Attention_RNN
from asr_arabic_transformer.attention.speech_model import CNN
from asr_arabic_transformer.utils import conv_output_shape, maxpool_output_shape


class SpeechRNN(nn.Module):
    def __init__(self, input_size, n_classes, Ty, project_size=25, encoder_hidden_size=128,
                 encoder_num_layers=2, decoder_hidden_size=128, decoder_num_layers=1, dropout=0.1, use_cnn=False):
        super().__init__()
        self.config = {'input_size': input_size, 'n_classes': n_classes, 'Ty': Ty, 'project_size': project_size,
                       'encoder_hidden_size': encoder_hidden_size, 'encoder_num_layers': encoder_num_layers,
                       'decoder_hidden_size': decoder_hidden_size,
                       'decoder_num_layers': decoder_num_layers, 'dropout': dropout, 'use_cnn': use_cnn}
        self.input_size = input_size
        self.n_classes = n_classes
        self.Ty = Ty

        self.cnn = None
        if use_cnn:
            self.cnn = CNN()

        if use_cnn:
            _, output_cnn = conv_output_shape((100, input_size), kernel_size=3, stride=2)
            _, output_cnn = conv_output_shape((100, output_cnn), kernel_size=3, stride=2)
            _, output_cnn = maxpool_output_shape((100, output_cnn), 2, 2)
            output_cnn *= self.cnn.conv2.out_channels
        else:
            output_cnn = input_size

        config = self.config.copy()
        config['num_alphabet'] = config['n_classes']
        config['input_size'] = output_cnn
        del config['n_classes']
        del config['use_cnn']
        self.attention_rnn = Attention_RNN(**config)
        self.use_cnn = use_cnn

    def forward(self, x):
        x = self._handle_cnn(x)
        return self.attention_rnn(x)

    def predict(self, x, eos, max_iter=100):
        x = self._handle_cnn(x)
        return self.attention_rnn.predict(x, eos, max_iter)

    def beam_search(self, x, beam_width, eos, max_iter=100):
        return self.attention_rnn.beam_search(x, beam_width, eos, max_iter=max_iter)

    def _handle_cnn(self, x):
        if self.use_cnn:
            x = x.unsqueeze(-3)
            x = torch.transpose(x, -1, -2)
            x = self.cnn(x)
            x = x.view(x.shape[0], -1, x.shape[-1])
            x = torch.transpose(x, -1, -2)
        return x


if __name__ == '__main__':
    model = SpeechRNN(10, 10, 20,
                      project_size=64, encoder_hidden_size=256,
                      encoder_num_layers=2, decoder_hidden_size=256, decoder_num_layers=1)
