import torch
from torch import nn
import torch.nn.functional as F


class Attention_RNN(nn.Module):
    """Class containing the architecture of the model and the corresponding weights
    Contains a classical implementation of attention model"""
    def __init__(self, input_size, num_alphabet, Ty, project_size=25, encoder_hidden_size=128,
                 encoder_num_layers=2, decoder_hidden_size=128, decoder_num_layers=1, save_attention=False):
        super().__init__()
        self.num_alphabet = num_alphabet
        self.input_size = input_size
        self.Ty = Ty
        self.Tx = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.fc1 = nn.Linear(input_size, project_size)
        self.encoder = nn.LSTM(project_size, hidden_size=encoder_hidden_size, num_layers=encoder_num_layers,
                               bidirectional=True, batch_first=True)
        self.post_attention_lstm = nn.LSTM(self.encoder.hidden_size * 2, hidden_size=decoder_hidden_size,
                                           num_layers=decoder_num_layers, batch_first=True)
        self.attention = nn.Sequential(nn.Linear(self.encoder.hidden_size*2 + self.post_attention_lstm.hidden_size, 80),
                                       nn.ReLU(), nn.Linear(80, 1))
        self.fc = nn.Linear(self.post_attention_lstm.hidden_size, self.num_alphabet)
        self.infos = {"attention" : []}
        self.save_attention = save_attention


    def calc_context(self, x, s_prev):
        repeat = s_prev.unsqueeze(1).expand(s_prev.shape[0], self.Tx, s_prev.shape[1])
        attention = torch.cat((x, repeat), dim=2)
        attention = self.attention(attention)
        attention = attention.squeeze(dim=-1)
        attention = F.softmax(attention, dim=1)
        context = (x * attention.unsqueeze(dim=-1)).sum(dim=1)
        if self.save_attention:
            self.infos["attention"].append(attention)
        return context

    def forward(self, x):
        if self.save_attention:
            self.infos["attention"] = []

        result = []
        x = self.fc1(x)
        x, _ = self.encoder(x)
        self.Tx = x.shape[1]
        s_prev, c_prev = self.init_hidden(self.post_attention_lstm, x.shape[0])
        for _ in range(self.Ty):
            context = self.calc_context(x, s_prev.squeeze(0))
            context = context.unsqueeze(dim=1)
            y, (s_prev, c_prev) = self.post_attention_lstm(context, (s_prev, c_prev))
            y = y.squeeze(dim=1)
            y = self.fc(y)
            result.append(y)
        result = torch.stack(result, dim=1)
        return result

    def init_hidden(self, layer, batch_size):
        h0 = torch.zeros((layer.num_layers, batch_size, layer.hidden_size)).to(self.device)
        c0 = torch.zeros((layer.num_layers, batch_size, layer.hidden_size)).to(self.device)
        return h0, c0

    def predict(self, x, eos, max_iter=100):
        if self.save_attention:
            self.infos["attention"] = []
        # (Tx, )
        result = []
        x = x.unsqueeze(0)
        x, _ = self.encoder(x)
        self.Tx = x.shape[1]
        s_prev, c_prev = self.init_hidden(self.post_attention_lstm, x.shape[0])
        for _ in range(max_iter):
            context = self.calc_context(x, s_prev.squeeze(0))
            context = context.unsqueeze(dim=1)
            y, (s_prev, c_prev) = self.post_attention_lstm(context, (s_prev, c_prev))
            y = y.squeeze(dim=1)
            y = self.fc(y)
            result.append(int(torch.argmax(y, -1)[0]))
            if result[-1] == eos:
                break
        result = torch.tensor(result)
        return result

if __name__ == "__main__":
    arch = Attention_RNN(num_alphabet=50, Ty=100, save_attention=True)
    x = torch.rand((5, 30, 50))
    y = arch(x)
    attention = arch.infos["attention"]
    print(attention[0].shape)
    attention = torch.stack(attention, dim=2)
    print(attention.shape)
