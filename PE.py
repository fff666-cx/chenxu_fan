import torch
import torch.nn as nn
import math
class Position_Encoding(nn.Module):
    def __init__(self,max_pos,dim):
        super(Position_Encoding,self).__init__()
        # pe为保存位置编码的数组,均初始化为0
        pe=torch.zeros(max_pos,dim)
        # 索引张量
        pos=torch.arange(0,max_pos).unsqueeze(1).float()
        # 生成间隔为2的序列，对应公式中的2i
        term=torch.arange(0,dim,2).float()
        term=torch.exp(term*(-math.log(10000.0)/dim))
        pe[:,0::2]=torch.sin(pos*term)
        pe[:,1::2]=torch.cos(pos*term)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x):
        return x+self.pe[:,:x.size(1)].clone().detach()