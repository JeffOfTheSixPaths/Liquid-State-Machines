import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
batch_size = 128
data_path = "./data/mnist"
num_steps = 25  # time steps
beta = 0.95     # LIF decay
lr = 1e-3

# Data transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

# Load MNIST
train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define Network
class SNN(nn.Module):
    def __init__(self):
        super().__init__()

        spike_grad = surrogate.fast_sigmoid()

        self.fc1 = nn.Linear(28*28, 1000)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(1000, 10)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)

        return torch.stack(spk2_rec)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SNN().to(device)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
def train():
    model.train()
    for epoch in range(1):  # increase for better accuracy
        for data, targets in train_loader:
            data = data.view(batch_size, -1).to(device)
            targets = targets.to(device)

            spk_rec = model(data)

            # Sum spikes over time
            spk_sum = spk_rec.sum(dim=0)

            loss = loss_fn(spk_sum, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} Loss: {loss.item():.4f}")

# Test loop
def test():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data = data.view(data.size(0), -1).to(device)
            targets = targets.to(device)

            spk_rec = model(data)
            spk_sum = spk_rec.sum(dim=0)

            _, predicted = spk_sum.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

train()
test()