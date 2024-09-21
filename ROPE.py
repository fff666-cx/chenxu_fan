import torch
import torch.nn as nn
import math

class RO_Positional_Encoding(nn.Module):
    def __init__(self, dim, max_pos):
        # dim为维度，max_pos为最大位置值
        super(RO_Positional_Encoding, self).__init__()
        self.max_pos = max_pos
        self.dim = dim
        self.register_buffer('sin_cos', self._generate_sin_cos(max_pos, dim))

    def _generate_sin_cos(self, max_pos, dim):
        # 调整为 (max_pos, 1) 的形状，索引张量
        position = torch.arange(0, max_pos).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        sin_cos = torch.cat((torch.sin(position * div_term), torch.cos(position * div_term)), dim=-1)
        return sin_cos

    def forward(self, x):
        seq_len, dim = x.size(1), x.size(2)
        # 确保输入张量的 dim 与实例变量 dim 一致。
        assert dim == self.dim
        pos = torch.arange(0, seq_len).unsqueeze(1).to(x.device)
        pos = self.sin_cos[pos % self.max_len]
        return x * pos[:, :dim].unsqueeze(1)