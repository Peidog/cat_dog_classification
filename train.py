import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 训练集和测试集路径
train_path = './Dataset/train'
test_path = './Dataset/test'

# 获取训练集和测试集集的标签
train_labels = sorted(os.listdir(train_path))
test_labels = sorted(os.listdir(test_path))
if train_labels == test_labels:
    animal_labels = train_labels = test_labels
print("animal_labels: ", animal_labels)

# 数据增强
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(148, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
])

test_transform = transforms.Compose([
    transforms.Resize((148, 148)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
])

# 加载数据集
train_dataset = ImageFolder(train_path, transform=train_transform)
test_dataset = ImageFolder(test_path, transform=test_transform)

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
class AnimalCNN(nn.Module):
    def __init__(self, num_classes):
        super(AnimalCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 9 * 9, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.pool(self.conv1(x))))
        x = self.relu(self.bn2(self.pool(self.conv2(x))))
        x = self.relu(self.bn3(self.pool(self.conv3(x))))
        x = self.relu(self.bn4(self.pool(self.conv4(x))))
        x = x.view(x.size(0), -1) 
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AnimalCNN(num_classes=len(animal_labels)).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 25 
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'Train Loss': running_loss / (pbar.n + 1),
                'Train Acc': correct / total
            })
            pbar.update(1)
    
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with tqdm(total=len(test_loader), desc='Testing', unit='batch') as pbar:
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                
                pbar.set_postfix({
                    'Test Loss': test_loss / (pbar.n + 1),
                    'Test Acc': correct / total
                })
                pbar.update(1)
    
    test_loss = test_loss / len(test_loader)
    test_acc = correct / total
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

    # 保存最佳模型
    if epoch == 0 or test_acc > max(test_accuracies[:-1]):
        torch.save(model.state_dict(), './weights/new_best_model.pth')

# 训练结束后，将结果写入文件
os.makedirs('./training_results', exist_ok=True)
# training_results = './training_results/training_results.txt'
training_results = './training_results/new_training_results.txt'
with open(training_results, 'w') as f:
    f.write('epoch,train_losses,train_accuracies,test_losses,test_accuracies\n')
    for i in range(len(train_losses)):
        f.write(f'{i+1},{train_losses[i]},{train_accuracies[i]},{test_losses[i]},{test_accuracies[i]}\n')

# 保存最终的模型
torch.save(model.state_dict(), './weights/new_final_model.pth')

