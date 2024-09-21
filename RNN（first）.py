import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
seed_all(seed)

'''定义一个simpleRNN 类，继承的nn.Module是PyTorch中所有神经网络的基类'''
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): # 传入输入层，隐藏层和输出层的大小
        # 继承父类的__init__函数
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # Linear layers for the RNN operations
        '''nn.Linear函数实现了线性变换，公式为y=xW^T+b，x为输入数据，W为权重矩阵，
        初始化nn.Linear时，权重使用标准正态分布初始化，偏置使用0进行初始化。b为偏置向量，y为输出数据'''
        '''self.input_to_hidden：将输入映射到隐藏状态。
           self.hidden_to_hidden：将前一隐藏状态映射到当前隐藏状态（处理循环）。
           self.hidden_to_output：将隐藏状态映射到输出。'''
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
    # 定义向前传播的过程
    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)
        # 两者分别为输入序列的批量大小和序列长度
        # Initialize hidden state
        hidden = torch.zeros(batch_size, self.hidden_size).to(x.device)
        '''初始化隐藏状态为零，形状为 (batch_size, hidden_size)，并将其移动到与输入数据相同的设备（CPU或GPU）。'''
        # Process each time step
        for t in range(seq_length):
            # Calculate new hidden state
            hidden = torch.tanh(self.input_to_hidden(x[:, t, :]) + self.hidden_to_hidden(hidden))
            # 用torch.tanh函数作为激活函数来计算隐藏状态
        # Calculate output from the final hidden state
        output = self.hidden_to_output(hidden)
        return output
# Define the weight initialization function
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# Data preparation
'''transforms.Compose：将多个数据变换操作组合成一个变换流程。
transforms.ToTensor()：将输入图像数据转换为 PyTorch 张量，并将像素值缩放到 [0, 1] 范围。
transforms.Normalize((0.5,), (0.5,))：对张量进行标准化。
这里是对单通道图像（灰度图）进行标准化，使其均值为 0.5，标准差为 0.5。这个标准化处理有助于加速训练并提高模型的收敛性。'''
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# 加载Fashion MINST 数据集（包含10类服装照片图片的数据集）root='./data'：指定数据集存储的位置。
# train=True：加载训练集（如果设置为 False，则加载测试集）。
# transform=transform：应用之前定义的数据变换。
# download=True：如果数据集不存在，则下载数据集。
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
'''DataLoader：创建一个数据加载器，用于批量加载数据。
dataset=train_dataset：指定要加载的数据集。
batch_size=64：每个批次包含 64 个样本。
shuffle=True：在每个 epoch 开始时打乱数据顺序，这有助于提高训练的泛化能力。'''
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Hyperparameters
'''超数据设置：input_size输入特征的维度，对于数据集来说，是一个28*28像素的灰度图，每个时间步的输入是一个长度为28的向量
hidden_size为128，代表隐藏层的大小，即隐藏单元的数量
output_size为输出特征的维度，代表输出的10个类别
其余两者分别为训练的轮数和学习率'''
input_size = 28
hidden_size = 128
output_size = 10
num_epochs = 18
learning_rate = 0.001

# Model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleRNN(input_size, hidden_size, output_size).to(device)
# Apply weight initialization
model.apply(initialize_weights)
# nn.CrossEntropyLoss()：定义损失函数。交叉熵损失函数通常用于分类问题，计算预测概率与真实标签之间的差异。
criterion = nn.CrossEntropyLoss()
# 定义优化器。Adam 优化器是一种自适应学习率优化算法，通过 model.parameters()
# 将模型的所有参数传递给优化器，并指定学习率 learning_rate。
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
train_losses = []
for epoch in range(num_epochs):
    epoch_loss = 0
    for images, labels in train_loader:
        images = images.view(-1, 28, 28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        # 计算预测输出与真实标签之间的损失值
        optimizer.zero_grad()
        # 清零优化器中的梯度，以防止梯度累加。
        loss.backward()
        # 执行反向传播，计算梯度
        optimizer.step()

        epoch_loss += loss.item()

    average_loss = epoch_loss / len(train_loader)
    train_losses.append(average_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
