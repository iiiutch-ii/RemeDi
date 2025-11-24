import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from optimizers.Muon_AdamW import MuonAdamW

# 1. 生成假数据 (1000个样本，每个样本10维特征)
torch.manual_seed(42)  # 固定随机种子保证可复现
input_size = 10
output_size = 1
num_samples = 1000

# 随机生成特征和标签（模拟回归任务）
X = torch.randn(num_samples, input_size)  # 1000x10的随机数据
y = X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(num_samples, 1)  # 线性关系+噪声

# 转换为PyTorch数据集
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. 定义DNN模型（两层全连接）
class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一层
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 第二层
        self.relu = nn.ReLU()  # 激活函数
        
    def forward(self, x):
        x = self.relu(self.fc1(x))  # 第一层+激活
        x = self.fc2(x)  # 第二层（无激活函数，适用于回归任务）
        return x

# 初始化模型
model = DNN(input_size, 64, output_size)  # 隐藏层维度设为64

# 3. 设置AdamW优化器和损失函数
optimizer = MuonAdamW([p for p in model.parameters()],lr = 0.01)
criterion = nn.MSELoss()  # 均方误差损失（回归任务）

# 4. 训练循环
num_epochs = 50
loss_history = []

model.train()  # 设置为训练模式
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()  # 梯度清零
        outputs = model(batch_X)  # 前向传播
        loss = criterion(outputs, batch_y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')





