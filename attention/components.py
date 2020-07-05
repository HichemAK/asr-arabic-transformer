import torch
from torch import nn
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.project = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.d_v = self.d_k

    def forward(self, q, k, v, mask=None):
        # (bs, sl, d_model)
        bs, sl, _ = q.shape
        q = self.q_linear(q).view(bs, sl, self.n_heads, self.d_k)
        k = self.k_linear(k).view(bs, sl, self.n_heads, self.d_k)
        v = self.v_linear(v).view(bs, sl, self.n_heads, self.d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        z = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(-3)
            z = z.masked_fill(mask == 0, -1e10)

        z = F.softmax(z, dim=-1) @ v
        z = self.project(z.transpose(1, 2).contiguous().view(bs, sl, -1))
        return z


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(self.size), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        p = self.pe[:, :seq_len].clone().detach()
        x = x + p
        x = self.dropout(x)
        return x

