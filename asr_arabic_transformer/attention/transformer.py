import torch
from torch import nn

from asr_arabic_transformer.attention.decoder import Decoder
from asr_arabic_transformer.attention.encoder import Encoder
from asr_arabic_transformer.utils import get_mask


class Transformer(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, n_heads=8, Ne=6, Nd=6, dropout=0.1, max_seq_len=512, device=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.Ne = Ne
        self.Nd = Nd
        self.dropout = dropout
        self.encoder = Encoder(d_model, d_ff, n_heads, Ne, dropout, max_seq_len)
        self.decoder = Decoder(d_model, d_ff, n_heads, Nd, dropout, max_seq_len)
        self.device = device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, src, target, src_mask=None, target_mask=None, src_padding=None, target_padding=None):
        """
        src : (N, S, D)
        target : (N, T, D)
        src_mask : (S, S)
        target_mask : (T, T)
        src_padding : (N, )
        target_padding : (N, )

        N : Batch size
        S : Sequence length of source
        T : Sequence length of target
        D : d_model of transformer
        """

        if src_padding is not None:
            if src_mask is None:
                src_mask = torch.ones((src.shape[-2], src.shape[-2]), dtype=torch.bool)
            src_mask_padding = torch.ones((src.shape[-3], src.shape[-2]), dtype=torch.bool)
            for i in range(target_padding.shape[0]):
                src_mask_padding[i, src_padding[i]:] = False
            src_mask = src_mask.unsqueeze(0)
            src_mask_padding = src_mask_padding.unsqueeze(1)
            src_mask = src_mask_padding & src_mask
            src_mask = ~src_mask

        if src_mask is not None and self.device == 'cuda':
            src_mask = src_mask.cuda()

        encoder_out = self.encoder(src, src_mask)
        if target_mask == 'triu':
            target_mask = get_mask(target.size(1))

        if target_padding is not None:
            if target_mask is None:
                target_mask = torch.zeros((target.shape[-2], target.shape[-2]), dtype=torch.bool)
            target_mask_padding = torch.ones((target.shape[-3], target.shape[-2]), dtype=torch.bool)
            for i in range(target_padding.shape[0]):
                target_mask_padding[i, target_padding[i]:] = False
            target_mask = ~target_mask
            target_mask = target_mask.unsqueeze(0)
            target_mask_padding = target_mask_padding.unsqueeze(1)
            target_mask = target_mask_padding & target_mask
            target_mask = ~target_mask

        if target_mask is not None and self.device == 'cuda':
            target_mask = target_mask.cuda()


        x = self.decoder(target, encoder_out, target_mask)
        return x


if __name__ == "__main__":
    from asr_arabic_transformer.utils import LabelSmoothLoss

    T = Transformer(d_model=10, d_ff=256, Ne=2, Nd=2, n_heads=2, max_seq_len=1000)
    criterion = LabelSmoothLoss(0.1)
    x = torch.rand((1, 1000, 10))
    target = torch.rand((1, 1000, 10))
    out = T(x, target[:, :-1, :])
    print(out.shape)
    loss = criterion(out[:, :target.size(1) - 1, :], target[:, 1:, :].argmax(dim=-1))
    print(loss.item())
