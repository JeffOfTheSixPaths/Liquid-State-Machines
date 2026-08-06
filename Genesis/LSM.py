import numpy as np
import tonic


SEED = 42
np.random.seed(SEED)

N_RESERVOIR = 500
TIMESTEPS = 100

N_TRAIN = 5000
N_TEST = 1000

LEAK = 0.95
THRESHOLD = 0.8
SPARSITY = 0.1
SPECTRAL_RADIUS = 0.9

RIDGE = 1.0  

INPUT_SIZE = 34 * 34 * 2

train_ds = tonic.datasets.NMNIST(save_to="./data", train=True)
test_ds = tonic.datasets.NMNIST(save_to="./data", train=False)

train_idx = np.random.permutation(len(train_ds))[:N_TRAIN]
test_idx = np.random.permutation(len(test_ds))[:N_TEST]

# print(type(test_ds))
# print(train_idx)
# print(type(train_idx[0]))

# for i in range(60):
#     _, label = train_ds[train_idx[i]]
#     print(train_idx[i], label)

# for i in range(60):
#     _, label = test_ds[test_idx[i]]
#     print(test_idx[i], label)

# print("============================================================")
# print(train_idx)


def events_to_spikes(events):
    spikes = np.zeros((TIMESTEPS, INPUT_SIZE), dtype=np.float32)

    if len(events) == 0:
        return spikes

    t0, t1 = events["t"].min(), events["t"].max()
    if t1 == t0:
        return spikes

    bins = ((events["t"] - t0) * (TIMESTEPS - 1) / (t1 - t0)).astype(np.int32)

    for x, y, p, b in zip(events["x"], events["y"], events["p"], bins):
        idx = int(y) * 34 + int(x)
        if p:
            idx += 34 * 34

        spikes[b, idx] += 1.0   # IMPORTANT: count events, don't overwrite

    return spikes


class LSM:
    def __init__(self):
        self.Win = np.random.randn(N_RESERVOIR, INPUT_SIZE).astype(np.float32) * 0.1

        W = np.random.randn(N_RESERVOIR, N_RESERVOIR).astype(np.float32)

        mask = np.random.rand(N_RESERVOIR, N_RESERVOIR) < SPARSITY
        W *= mask

        self.Wmask = W.copy()

        eig = np.max(np.abs(np.linalg.eigvals(W)))
        W *= SPECTRAL_RADIUS / (eig + 1e-8)

        self.W = W.astype(np.float32)

    def run(self, spike_train):
        v = np.zeros(N_RESERVOIR, dtype=np.float32)
        spikes = np.zeros(N_RESERVOIR, dtype=np.float32)

        states = []

        for inp in spike_train:
            l = LEAK * v
            winput = self.Win @ inp
            wspikes = self.W @ spikes
            v =  l + winput + wspikes
            spikes = (v > THRESHOLD).astype(np.float32)
            v[spikes > 0] = 0.0

            states.append(spikes.copy())

        return np.array(states)

    def add_node(self):
        global N_RESERVOIR
        N_RESERVOIR += 1
        W_new = np.random.randn(N_RESERVOIR, N_RESERVOIR).astype(np.float32)
        
        # need to copy the values from self.W into W_new 
        for i in range(N_RESERVOIR - 1):
            for j in range(N_RESERVOIR - 1):
                W_new[i, j] = self.W[i, j]

        new_win = self.Win = np.random.randn(N_RESERVOIR, INPUT_SIZE).astype(np.float32) * 0.1
        for i in range(N_RESERVOIR - 1):
            for j in range(N_RESERVOIR - 1):
                new_win[i, j] = self.Win[i, j]
        self.Win = new_win.copy()
        
        # make sparse
        mask = np.random.rand(N_RESERVOIR, N_RESERVOIR) < SPARSITY  # only apply to new synapses
        # print("mask")
        # print(mask)
        for i in range(N_RESERVOIR - 1):
            W_new[N_RESERVOIR - 1, i] *= mask[N_RESERVOIR - 1, i]
        
        # print("W_new after bottom row: ")
        # print(W_new)

        for i in range(N_RESERVOIR - 1):
            W_new[i, N_RESERVOIR - 1] *= mask[i, N_RESERVOIR - 1]
        # print("W_new after side column: ")
        # print(W_new)
        
    
        # TODO: need to do spectral radius later

        self.W = W_new.copy()





lsm = LSM()


def extract(events):
    spikes = events_to_spikes(events)
    states = lsm.run(spikes)

    mean_rate = states.mean(axis=0)
    final_state = states[-1]

    feat = np.concatenate([mean_rate, final_state])

    # normalization helps conditioning a LOT
    feat = feat / (np.linalg.norm(feat) + 1e-8)

    return feat


history_X = []
history_Y = []

num_samples_per_episode = 10
window = 6
threshold_n = 1
prev_scores = []
Dp = 0 # \Delta P
dp = 0 # \delta P
Dp_hist = []
dp_hist = []


batch = train_idx[0: num_samples_per_episode]
for idx in batch:
    events, label = train_ds[idx]
    feat = extract(events)
    history_X.append(feat)
    history_Y.append(label)

for start in range(num_samples_per_episode + 1, len(train_idx) - 1 , num_samples_per_episode):
    batch = train_idx[start: start + num_samples_per_episode]
    for idx in batch:
        events, label = train_ds[idx]

        feat = extract(events)
        history_X.append(feat)
        history_Y.append(label)

    x_train = np.array(history_X[:-num_samples_per_episode])
    y_train = np.array(history_Y[:-num_samples_per_episode]) # except the last thing
    # one hot encoding
    Y = np.zeros((len(y_train), 10), dtype=np.float32)
    Y[np.arange(len(y_train)), y_train] = 1.0

    #train readout on everything except last "episode"
    A = x_train.T @ x_train
    B = x_train.T @ Y
    Wout = np.linalg.solve(A + RIDGE * np.eye(A.shape[0]), B)

    # check threshold 
    
    # run everything on the new episode
    curr_score = 1
    correct = 0
    for i, idx  in enumerate(train_idx[start: start + num_samples_per_episode]):
        events, label = train_ds[idx]

        feat = extract(events)
        pred = np.argmax(feat @ Wout)

        correct += (pred == label)
        curr_score = correct

    prev_scores.append(curr_score)

    if len(prev_scores) < window: 
        continue

    prev_score = prev_scores[-window]
    Dp = (curr_score - prev_score) / window
    Dp_hist.append(Dp)
    

    e_i = (start - window*num_samples_per_episode) //num_samples_per_episode
    dp = 0
    for i in range(e_i, e_i + window):
        dp += prev_scores[i]
    dp /= window*num_samples_per_episode
    dp_hist.append(dp)

    # print(dp , threshold_n)
    # if dp < threshold_n:
    #     lsm.add_node()
    #     print("added lsm node with dp: ", dp)

    if len(prev_scores) % 50 == 0 and i > 0:
        print(np.mean(np.asarray(prev_scores)))


# #print(x_train)
# print("Evaluating...")

# correct = 0

# for i, idx in enumerate(test_idx):
#     events, label = test_ds[idx]

#     feat = extract(events)
#     pred = np.argmax(feat @ Wout)

#     correct += (pred == label)

#     if i % 500 == 0 and i > 0:
#         print(i, correct / i)

# print("\nFinal Accuracy:", correct / len(test_idx))

import matplotlib.pyplot as plt

print(Dp_hist)
print(dp_hist)
plt.plot(prev_scores)
#plt.plot([x/num_samples_per_episode for x in prev_scores])
#plt.plot(Dp_hist)
#plt.plot(dp_hist)
plt.xlabel("Episode")
plt.ylabel("Accuracy")
plt.title("Episode Accuracy")
plt.show()







    

  








exit() # beyond here is normal tranining




print("Extracting training features...")

history_X = []
y_train = []

for i, idx in enumerate(train_idx):
    events, label = train_ds[idx]

    history_X.append(extract(events))
    y_train.append(label)

    if i % 500 == 0:
        print("train:", i)

history_X = np.array(history_X)
y_train = np.array(y_train)


Y = np.zeros((len(y_train), 10), dtype=np.float32)
Y[np.arange(len(y_train)), y_train] = 1.0

print("Training readout...")

A = history_X.T @ history_X
B = history_X.T @ Y

Wout = np.linalg.solve(A + RIDGE * np.eye(A.shape[0]), B)


print(history_X)
print("Evaluating...")

correct = 0

for i, idx in enumerate(test_idx):
    events, label = test_ds[idx]

    feat = extract(events)
    pred = np.argmax(feat @ Wout)

    correct += (pred == label)

    if i % 500 == 0 and i > 0:
        print(i, correct / i)

print("\nFinal Accuracy:", correct / len(test_idx))
