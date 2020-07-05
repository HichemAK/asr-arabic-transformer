import torch
from torch import nn
from attention.transformer import Transformer


class SpeechModel(nn.Module):
    def __init__(self, input_size, n_classes, **transformer_params):
        super().__init__()
        self.n_classes = n_classes
        self.transformer = Transformer(**transformer_params)
        self.linear1 = nn.Linear(input_size, self.transformer.d_model)
        self.embed = nn.Embedding(n_classes, self.transformer.d_model)
        self.linear2 = nn.Linear(self.transformer.d_model, n_classes)

    def forward(self, src, target):
        src = self.linear1(src)
        target = self.embed(target)
        x = self.transformer(src, target)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    batch_size = 8
    input_seq_length = 200
    output_seq_length = 30
    input_size = 100
    n_classes = 50


    T = SpeechModel(input_size=input_size, n_classes=n_classes ,d_model=256, d_ff=512, Ne=4, Nd=2, n_heads=2, max_seq_len=512)
    x = torch.rand((batch_size, input_seq_length, input_size))
    target = torch.randint(n_classes, (batch_size, output_seq_length))

    res = T(x, target)
    print(res.shape)

    total_params = sum(p.numel() for p in T.parameters())
    total_trainable_params = sum(p.numel() for p in T.parameters() if p.requires_grad)
    print(total_params, total_trainable_params)
