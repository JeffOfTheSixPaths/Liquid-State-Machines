import torch
import torch.nn as nn
import snntorch as snn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================
# Hyperparameters
# =========================
batch_size = 64
time_steps = 25
input_size = 28 * 28
reservoir_size = 500
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
# LSM Reservoir (no readout)
# =========================
class Reservoir(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_fc = nn.Linear(input_size, reservoir_size, bias=False)
        self.reservoir_fc = nn.Linear(reservoir_size, reservoir_size, bias=False)
        self.lif = snn.Leaky(beta=beta)

        # Initialize sparse reservoir
        with torch.no_grad():
            mask = (torch.rand_like(self.reservoir_fc.weight) < 0.1).float()
            self.reservoir_fc.weight *= mask
            self.reservoir_fc.weight *= 0.5

        # Freeze weights
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, spike_input):
        mem = self.lif.init_leaky()
        spk = torch.zeros(spike_input.size(1), reservoir_size, device=spike_input.device)

        spike_record = []

        for t in range(spike_input.size(0)):
            input_current = self.input_fc(spike_input[t])
            recurrent_current = self.reservoir_fc(spk)

            spk, mem = self.lif(input_current + recurrent_current, mem)
            spike_record.append(spk)

        # Return time-aggregated state
        return torch.stack(spike_record).sum(dim=0)

# =========================
# Build Reservoir
# =========================
reservoir = Reservoir().to(device)

# =========================
# Collect Features (X) and Labels (Y)
# =========================
X_list = []
Y_list = []

reservoir.eval()

with torch.no_grad():
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)

        spike_data = rate_encode(data, time_steps)
        features = reservoir(spike_data)  # [batch, reservoir_size]

        X_list.append(features)
        
        # One-hot encode labels
        y_onehot = torch.zeros(targets.size(0), output_size, device=device)
        y_onehot.scatter_(1, targets.unsqueeze(1), 1.0)
        Y_list.append(y_onehot)

# Stack all data
X = torch.cat(X_list, dim=0)  # [N, reservoir_size]
Y = torch.cat(Y_list, dim=0)  # [N, output_size]

# =========================
# Solve Linear Regression
# =========================
# Solve XW = Y  →  W
W = torch.linalg.lstsq(X, Y).solution  # [reservoir_size, output_size]

# =========================
# Evaluation
# =========================
correct = 0
total = 0

with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)

        spike_data = rate_encode(data, time_steps)
        features = reservoir(spike_data)

        outputs = features @ W
        preds = outputs.argmax(dim=1)

        correct += (preds == targets).sum().item()
        total += targets.size(0)

print("Test Accuracy:", correct / total)