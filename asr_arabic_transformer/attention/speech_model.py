import torch
from torch import nn
from asr_arabic_transformer.attention.transformer import Transformer
from asr_arabic_transformer.attention.transformer_torch import TransformerTorch
from asr_arabic_transformer.utils import conv_output_shape, maxpool_output_shape, get_mask
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2))

        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        x = self.batch_norm1(F.relu(self.conv1(x)))
        x = self.batch_norm2(F.relu(self.conv2(x)))
        x = self.max_pool(x)
        return x


class SpeechModel(nn.Module):
    def __init__(self, input_size, n_classes, use_cnn=True, torch_version=False, **transformer_params):
        super().__init__()
        self.cnn = None
        if use_cnn:
            self.cnn = CNN()
        self.n_classes = n_classes
        if use_cnn:
            _, output_cnn = conv_output_shape((100, input_size), kernel_size=3, stride=2)
            _, output_cnn= conv_output_shape((100, output_cnn), kernel_size=3, stride=2)
            _, output_cnn = maxpool_output_shape((100, output_cnn), 2, 2)
            output_cnn *= self.cnn.conv2.out_channels
        else:
            output_cnn = input_size
        if torch_version:
            self.transformer = TransformerTorch(**transformer_params)
        else:
            self.transformer = Transformer(**transformer_params)

        self.linear1 = nn.Linear(output_cnn, self.transformer.d_model)
        self.embed = nn.Embedding(n_classes, self.transformer.d_model)
        self.linear2 = nn.Linear(self.transformer.d_model, n_classes)
        self.use_cnn = use_cnn

    def forward(self, src, target, src_padding, target_padding):
        if self.use_cnn:
            src = src.unsqueeze(-3)
            src = torch.transpose(src, -1, -2)
            src = self.cnn(src)
            src = src.view(src.shape[0], -1, src.shape[-1])
            src = torch.transpose(src, -1, -2)
            src_padding = None

        src = self.linear1(src)
        target = self.embed(target)
        x = self.transformer(src, target, target_mask='triu', src_padding=src_padding, target_padding=target_padding)
        x = self.linear2(x)
        return x

    def predict(self, x, sos, eos):
        x = x.unsqueeze(-3).unsqueeze(0)
        x = torch.transpose(x, -1, -2)
        x = self.cnn(x)
        x = x.view(x.shape[0], -1, x.shape[-1])

        x = torch.transpose(x, -1, -2)
        x = self.linear1(x)

        x = self.transformer.encoder(x)
        outputs = [sos, ]
        last_output = -1
        while last_output != eos:
            target = torch.tensor(outputs).unsqueeze(0)
            target = self.embed(target)
            target_mask = get_mask(target.shape[-2])
            out = self.transformer.decoder(target, x, target_mask)
            out = self.linear2(out)
            last_output = int(torch.argmax(out, dim=-1)[0, -1])
            outputs.append(last_output)
        return outputs


if __name__ == "__main__":
    batch_size = 8
    input_seq_length = 200
    output_seq_length = 30
    input_size = 100
    n_classes = 5

    T = SpeechModel(input_size=input_size, n_classes=n_classes, d_model=256, d_ff=512, Ne=4, Nd=2, n_heads=2,
                    max_seq_len=512)
    x = torch.rand((batch_size, input_seq_length, input_size))
    target = torch.randint(n_classes, (batch_size, output_seq_length))

    res = T(x, target)
    print(res.shape)

    total_params = sum(p.numel() for p in T.parameters())
    total_trainable_params = sum(p.numel() for p in T.parameters() if p.requires_grad)
    print(total_params, total_trainable_params)

    x = torch.rand((input_size, input_seq_length))
    y = T.predict(x, sos=1, eos=2)
