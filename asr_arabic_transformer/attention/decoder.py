from torch import nn
from torch.nn import ModuleList

from asr_arabic_transformer.attention.components import MultiHeadAttention, Norm, FeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout):
        super().__init__()

        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.norm3 = Norm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.masked_multi_head = MultiHeadAttention(d_model, n_heads)
        self.encoder_decoder_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, x, encoder_out, mask=None):
        x = self.norm1(x + self.dropout1(self.masked_multi_head(x, x, x, mask)))
        x = self.norm2(x + self.dropout2(self.encoder_decoder_attention(x, encoder_out, encoder_out)))
        x = self.norm3(x + self.dropout3(self.feed_forward(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, N, dropout=0.1):
        super().__init__()
        self.decoder_layers = ModuleList([DecoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(N)])

    def forward(self, target, encoder_out, target_mask=None):
        x = target
        for dec in self.decoder_layers:
            x = dec(x, encoder_out, target_mask)
        return x
