import torch
from torch import nn
from torch.nn.modules.transformer import Transformer

from asr_arabic_transformer.utils import get_mask


class TransformerTorch(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, n_heads=8, Ne=6, Nd=6, dropout=0.1, device=None):
        super().__init__()
        self.transformer = Transformer(d_model=d_model, nhead=n_heads, num_encoder_layers=Ne, num_decoder_layers=Nd,
                                       dim_feedforward=d_ff, dropout=dropout)
        self.device = device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.d_model = d_model

    def forward(self, src, target, src_mask=None, target_mask=None, src_padding=None, target_padding=None):
        if target_mask == 'triu':
            target_mask = get_mask(target.size(1))
        src_mask_padding = None
        target_mask_padding = None
        if src_padding is not None:
            src_mask_padding = torch.zeros((src.shape[-3], src.shape[-2]), dtype=torch.bool)
            for i in range(target_padding.shape[0]):
                src_mask_padding[i, src_padding[i]:] = True

        if target_padding is not None:
            target_mask_padding = torch.zeros((target.shape[-3], target.shape[-2]), dtype=torch.bool)
            for i in range(target_padding.shape[0]):
                target_mask_padding[i, target_padding[i]:] = True

        if self.device == 'cuda':
            target_mask = target_mask.cuda() if target_mask is not None else None
            src_mask_padding = src_mask_padding.cuda() if src_mask_padding is not None else None
            target_mask_padding = target_mask_padding.cuda() if target_mask_padding is not None else None

        src = torch.transpose(src, 0, 1)
        target = torch.transpose(target, 0, 1)
        res = self.transformer(src, target, src_mask, target_mask, None, src_mask_padding, target_mask_padding)
        res = res.transpose(0, 1)
        return res
