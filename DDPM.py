import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
class NumpyDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)  # 转换为 PyTorch tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
# 数据准备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 选择标签为1的图像
selected_digits = [1]
train_images = [img for img, label in mnist if label in selected_digits]
train_images = np.array(train_images)
train_images=NumpyDataset(train_images)
train_size = int(0.8 * len(train_images))
val_size = len(train_images) - train_size
train_dataset, val_dataset = random_split(train_images, [train_size, val_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)


# RNN模型定义
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)
        hidden = torch.zeros(batch_size, self.hidden_size).to(x.device)
        for t in range(seq_length):
            input_to_hidden_output = self.input_to_hidden(x[:, t, :])
            hidden = torch.tanh(input_to_hidden_output + self.hidden_to_hidden(hidden))

        output = self.hidden_to_output(hidden)
        return output
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# 扩散过程
def forward_diffusion(x_0, t, beta):
    noise = torch.randn_like(x_0)
    return torch.sqrt(1 - beta[t]) * x_0 + torch.sqrt(beta[t]) * noise
# 训练模型
epoch=0
class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.stopped_epoch = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss - val_loss > self.delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save the model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopped_epoch = epoch
early_stopping = EarlyStopping(patience=5, delta=0.01)


def train(model, train_loader, val_loader, optimizer, num_epochs, timesteps, beta, device):
    loss_fn = nn.MSELoss()
    train_losses = []
    val_losses = []

    model.to(device)

    for epoch in range(num_epochs):
        model.train()  # 切换到训练模式
        for batch in train_loader:
            batch = batch.view(batch.size(0), -1, 28).to(device)  # 调整输入形状并移动到设备
            optimizer.zero_grad()
            noise = forward_diffusion(batch, torch.randint(0, timesteps, (1,)).item(), beta)
            predictions = model(noise)
            train_loss = loss_fn(predictions, batch[:,-1,:])
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())

        model.eval()  # 切换到评估模式
        with torch.no_grad():  
            for batch in val_loader:
                batch = batch.view(batch.size(0), -1, 28).to(device)  # 调整输入形状并移动到设备
                noise = forward_diffusion(batch, torch.randint(0, timesteps, (1,)).item(), beta)
                predictions = model(noise)
                val_loss = loss_fn(predictions, batch[:,-1,:])
                val_losses.append(val_loss.item())

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return train_losses, val_losses



# 生成图像

def generate_images(model, timesteps, beta, device, num_images=10):
    model.eval()  # 切换到评估模式
    generated_images = []

    # 初始化随机噪声
    noise = torch.randn(num_images, 1, 28).to(device)  # 输入序列的长度为1，28为特征维度

    for t in reversed(range(timesteps)):
        with torch.no_grad():
            # 将当前噪声输入到模型中
            predictions = model(noise)
            # 反向扩散过程：根据模型的输出和当前噪声更新噪声
            noise = reverse_diffusion(noise, predictions, t, beta)


    generated_images.append(noise.cpu().numpy())

# 绘制生成的图像
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
       ax = axes[i]
       ax.imshow(generated_images[0][i].squeeze(), cmap='gray')
       ax.axis('off')
    plt.show()

def reverse_diffusion(noise, predictions, t, beta):
    alpha_t = 1 - beta[t]
    return (noise - (beta[t] * predictions)) / torch.sqrt(alpha_t)
# 超参数设置
input_size = 28  # 图像宽度
hidden_size = 128  # RNN隐藏层大小
output_size = 28  # 输出图像宽度
num_epochs = 10
timesteps = 50  # 扩散步数
beta = torch.linspace(0.1, 0.2, timesteps)  # beta设置

# 初始化模型和优化器
model = SimpleRNN(input_size, hidden_size, output_size)
model.apply(initialize_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train_losses,val_losses=train(model, train_loader,val_loader, optimizer, num_epochs, timesteps, beta,device)
# 绘制训练损失图
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Iterations')
plt.ylabel('train_Loss')
plt.show()
plt.plot(val_losses)
plt.title('Val Loss')
plt.xlabel('Iterations')
plt.ylabel('val_Loss')
plt.show()
# 生成和展示图像
generate_images(model, timesteps, beta, device, num_images=10)
