from attention.decoder import Decoder
from attention.encoder import Encoder
from torch import nn
import torch
from utils import normalize_length

class Transformer(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, n_heads=8, Ne=6, Nd=6, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.Ne = Ne
        self.Nd = Nd
        self.dropout = dropout
        self.encoder = Encoder(d_model, d_ff, n_heads, Ne, dropout, max_seq_len)
        self.decoder = Decoder(d_model, d_ff, n_heads, Nd, dropout, max_seq_len)

    def forward(self, src, target, src_mask=None, target_mask=None):
        encoder_out = self.encoder(src, src_mask)
        encoder_out, target, target_mask = normalize_length(encoder_out, target)
        print(encoder_out.shape, target.shape, target_mask.shape)
        x = self.decoder(target, encoder_out, target_mask)
        return x


if __name__ == "__main__":
    T = Transformer(d_model=10, d_ff=256, Ne=2, Nd=2, n_heads=2, max_seq_len=20)
    x = torch.rand((1, 1, 10))
    target = torch.rand((1, 2, 10))
    out = T(x, target)
    print(out.shape)
