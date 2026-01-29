import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

#######################################
# Configuration
#######################################
device = "cpu"
torch.manual_seed(0)
np.random.seed(0)

TIME_STEPS = 25
N_INPUT = 28 * 28
N_RES = 1000
BATCH_SIZE = 1

#######################################
# Dataset (MNIST)
#######################################
transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = datasets.MNIST(root=".", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root=".", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# class LSM(nn.Module):
#     def __init__(self, res_sparsity=0.1, in_sparsity=0.2):
#         super().__init__()

#         self.fc_in = nn.Linear(N_INPUT, N_RES, bias=False)
#         self.fc_rec = nn.Linear(N_RES, N_RES, bias=False)

#         self.lif = snn.Leaky(
#             beta=0.95,
#             spike_grad=surrogate.fast_sigmoid()
#         )

#         # Initialize weights
#         nn.init.normal_(self.fc_in.weight, mean=0.0, std=0.3)
#         nn.init.normal_(self.fc_rec.weight, mean=0.0, std=0.1)

#         ################################
#         # Create sparse masks
#         ################################

#         # Reservoir mask
#         rec_mask = (torch.rand(N_RES, N_RES) < res_sparsity).float()

#         # Remove self-connections (optional but typical)
#         rec_mask.fill_diagonal_(0)

#         # Input mask (optional)
#         in_mask = (torch.rand(N_RES, N_INPUT) < in_sparsity).float()

#         ################################
#         # Apply masks
#         ################################

#         with torch.no_grad():
#             self.fc_rec.weight *= rec_mask
#             self.fc_in.weight *= in_mask

#             # Scale recurrent weights for stability
#             spectral_radius = torch.max(
#                 torch.abs(torch.linalg.eigvals(self.fc_rec.weight))
#             ).real

#             self.fc_rec.weight *= 0.9 / spectral_radius

#         ################################
#         # Freeze reservoir
#         ################################

#         for p in self.parameters():
#             p.requires_grad = False

#         # Store masks (not strictly needed, but useful)
#         self.register_buffer("rec_mask", rec_mask)
#         self.register_buffer("in_mask", in_mask)

#     def forward(self, x):
#         mem = torch.zeros((x.size(0), N_RES), device=x.device)
#         spk = torch.zeros((x.size(0), N_RES), device=x.device)

#         spike_sum = torch.zeros((x.size(0), N_RES), device=x.device)

#         for _ in range(TIME_STEPS):
#             cur = self.fc_in(x) + self.fc_rec(spk)
#             spk, mem = self.lif(cur, mem)
#             spike_sum += spk

#         return spike_sum / TIME_STEPS


class LSM(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc_in = nn.Linear(N_INPUT, N_RES, bias=False)
        self.fc_rec = nn.Linear(N_RES, N_RES, bias=False)

        # Freeze reservoir weights
        for p in self.parameters():
            p.requires_grad = False

        self.lif = snn.Leaky(
            beta=0.95,
            spike_grad=surrogate.fast_sigmoid()
        )

        nn.init.normal_(self.fc_in.weight, mean=0.0, std=0.3)
        nn.init.normal_(self.fc_rec.weight, mean=0.0, std=0.1)

    def forward(self, x): 
        mem = torch.zeros((x.size(0), N_RES), device=x.device)
        spk = torch.zeros((x.size(0), N_RES), device=x.device)

        spike_sum = torch.zeros((x.size(0), N_RES), device=x.device)

        for _ in range(TIME_STEPS):
            cur = self.fc_in(x) + self.fc_rec(spk)
            spk, mem = self.lif(cur, mem)
            spike_sum += spk

        return spike_sum / TIME_STEPS


lsm = LSM().to(device)

#######################################
# Collect reservoir states
#######################################
def collect_states(loader, max_samples):
    states = []
    labels = []

    for i, (data, target) in enumerate(loader):
        if i >= max_samples:
            break

        x = data.view(1, -1).to(device)
        state = lsm(x).detach().cpu().numpy()[0]

        states.append(state)
        labels.append(target.item())

    return np.array(states), np.array(labels)


print("Extracting training states...")
X_train, y_train = collect_states(train_loader, max_samples=6000)

print("Extracting test states...")
X_test, y_test = collect_states(test_loader, max_samples=1000)

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
