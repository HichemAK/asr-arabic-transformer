from torch import nn
from torch.nn import ModuleList

from asr_arabic_transformer.attention.components import MultiHeadAttention, Norm, FeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout):
        super().__init__()

        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.multi_head = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout1(self.multi_head(x, x, x, mask)))
        x = self.norm2(x + self.dropout2(self.feed_forward(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, N, dropout=0.1):
        super().__init__()
        self.encoder_layers = ModuleList([EncoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(N)])

    def forward(self, x, mask=None):
        for enc in self.encoder_layers: # SUSPECT
            x = enc(x, mask)
        return x
