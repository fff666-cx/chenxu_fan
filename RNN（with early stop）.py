import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import random
# 设置随机种子
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
seed_all(seed)

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
            hidden = torch.tanh(self.input_to_hidden(x[:, t, :]) + self.hidden_to_hidden(hidden))
        output = self.hidden_to_output(hidden)
        return output

# 对权重进行新的可视化
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

input_size = 28
hidden_size = 128
output_size = 10
num_epochs = 20
learning_rate = 0.001


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleRNN(input_size, hidden_size, output_size).to(device)
model.apply(initialize_weights)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 设置早停机制
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
# 计算准确率
def calculate_accuracy(loader, model, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.view(-1, 28, 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

early_stopping = EarlyStopping(patience=5, delta=0.01)


train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, labels in train_loader:
        images = images.view(-1, 28, 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    average_train_loss = epoch_loss / len(train_loader)
    train_losses.append(average_train_loss)
    train_accuracy = calculate_accuracy(train_loader, model, device)
    train_accuracies.append(train_accuracy)

    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.view(-1, 28, 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    average_val_loss = val_loss / len(val_loader)
    val_losses.append(average_val_loss)
    val_accuracy = calculate_accuracy(val_loader, model, device)
    val_accuracies.append(val_accuracy)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {average_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
    print(f'Validation Loss: {average_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    early_stopping(average_val_loss, model)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch + 1}")
        break


model.load_state_dict(torch.load('best_model.pth'))


plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 部分图像的分类可视化
def visualize_predictions(model, data_loader, device):
    model.eval()
    images, labels = next(iter(data_loader))
    images = images.view(-1, 28, 28).to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    
    images = images.cpu().numpy()
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()

    
    fig, axes = plt.subplots(1, 10, figsize=(15, 3))
    for i in range(10):
        ax = axes[i]
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f'Pred: {predicted[i]} \nTrue: {labels[i]}')
        ax.axis('off')

    plt.show()



visualize_predictions(model, val_loader, device)
