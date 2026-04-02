import torch
import torch.nn as nn
import snntorch as snn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# =========================
# Hyperparameters
# =========================
batch_size = 32
time_steps = 25
input_size = 28 * 28
reservoir_size = 2000
output_size = 10
beta = 0.9
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Data
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

train_dataset = datasets.MNIST(".", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(".", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# =========================
# Rate Encoding
# =========================
def rate_encode(x, time_steps):
    x = x.view(x.size(0), -1)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    spikes = torch.rand(time_steps, *x.shape, device=x.device) < x
    return spikes.float()

# =========================
# LSM Model
# =========================
class LSM(nn.Module):
    def __init__(self):
        super().__init__()

        # Input → reservoir
        self.input_fc = nn.Linear(input_size, reservoir_size, bias=False)

        # Recurrent reservoir (sparse)
        self.reservoir_fc = nn.Linear(reservoir_size, reservoir_size, bias=False)

        # LIF neurons
        self.lif = snn.Leaky(beta=beta)

        # Readout (ONLY trainable part)
        self.readout = nn.Linear(reservoir_size, output_size)

        # ---- Initialize reservoir ----
        with torch.no_grad():
            # Sparse connectivity
            mask = (torch.rand_like(self.reservoir_fc.weight) < 0.1).float()
            self.reservoir_fc.weight *= mask

            # Scale weights (stability control)
            self.reservoir_fc.weight *= 0.5

        # Freeze reservoir
        for p in self.input_fc.parameters():
            p.requires_grad = False
        for p in self.reservoir_fc.parameters():
            p.requires_grad = False

    def forward(self, spike_input):
        mem = self.lif.init_leaky()
        spk = torch.zeros(spike_input.size(1), reservoir_size, device=spike_input.device)

        spike_record = []

        for t in range(spike_input.size(0)):
            input_current = self.input_fc(spike_input[t])
            recurrent_current = self.reservoir_fc(spk)

            total_current = input_current + recurrent_current

            spk, mem = self.lif(total_current, mem)
            spike_record.append(spk)

        # Sum spikes over time (rate-based readout)
        spike_sum = torch.stack(spike_record).sum(dim=0)

        return self.readout(spike_sum)

# =========================
# Setup
# =========================

lsm = LSM().to(device)

def collect_states(loader, max_samples):
    states = []
    labels = []

    for i, (data, target) in enumerate(loader):
        if i >= max_samples:
            break

        x = data.view(1, -1).to(device)
        print(x.shape)
        state = lsm(x).detach().cpu().numpy()[0]

        states.append(state)
        labels.append(target.item())

    return np.array(states), np.array(labels)

start = time.time()
print("Extracting training states...")
X_train, y_train = collect_states(train_loader, max_samples=6000)
print(time.time() - start)

print("Extracting test states...")
X_test, y_test = collect_states(test_loader, max_samples=1000)
print(time.time() - start)


#######################################
# Train linear readout
#######################################
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = LogisticRegression(
    max_iter=2000
)

clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)

print(f"\n✅ Test accuracy: {acc:.3f}")



model = LSM().to(device)
optimizer = torch.optim.Adam(model.readout.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# =========================
# Training
# =========================
for epoch in range(3):
    model.train()
    total_loss = 0

    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)

        spike_data = rate_encode(data, time_steps)

        optimizer.zero_grad()
        output = model(spike_data)
        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")

# =========================
# Evaluation
# =========================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        spike_data = rate_encode(data, time_steps)

        output = model(spike_data)
        preds = output.argmax(dim=1)

        correct += (preds == targets).sum().item()
        total += targets.size(0)

print("Test Accuracy:", correct / total)


