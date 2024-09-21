import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        # 确保嵌入维度可以被头数整除
        assert embed_size % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.embed_size = embed_size

        # 定义生成Q,K,V的线性层
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        # 将多头注意力的输出映射回原始的嵌入维度
        self.out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        # 获取输入的批次大小和序列长度
        N = x.shape[0]
        seq_length = x.shape[1]

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # 调整维度为(batch_size, seq_length, num_heads, head_dim)方便进行多头处理
        queries = queries.view(N, seq_length, self.num_heads, self.head_dim)
        keys = keys.view(N, seq_length, self.num_heads, self.head_dim)
        values = values.view(N, seq_length, self.num_heads, self.head_dim)
        # 调整维度顺序以便进行注意力计算：(batch_size, num_heads, seq_length, head_dim)
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        # 计算缩放点积注意力分数，并应用 softmax 得到注意力权重。
        attention_scores = torch.matmul(queries, keys.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        # 合并多个头的输出，并调整维度回到原始形状 (batch_size, seq_length, embed_size)
        attention_output = torch.matmul(attention_weights, values)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(N, seq_length, self.embed_size)

        output = self.out(attention_output)

        return output
# 设置随机种子
torch.manual_seed(0)

# 定义超参数
embed_size = 128  # 嵌入维度
num_heads = 8    # 头数
batch_size = 16  # 批次大小
seq_length = 30  # 序列长度

# 生成一个（10，20，64）的随机张量
x = torch.rand(batch_size, seq_length, embed_size)


attention_layer = MultiHeadAttention(embed_size, num_heads)


output = attention_layer(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print(x)
print(output)
