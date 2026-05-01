from gensim.models import FastText # need to cite this paper
import snntorch as snn
from snntorch import spikegen
import torch
import torch.nn as nn
import numpy as np
import random


class Neuron(): # need to add __str__ 
    def __init__(self, beta = 0.9):
        self.neuron = snn.Leaky(beta = 0.9)
        self.connections = []
        self.weights = []
        self.history = []
        self.input_spikes = torch.tensor(0.0)
        self.spike = torch.tensor(0.0)
        self.mem = torch.tensor(0.0)

    def add_connection(self, n, weight = 0.7, randomize_weight = True): # Randomize weight overrides the given weight, if any
        self.connections.append(n)
        self.weights.append(np.random.random() if randomize_weight else weight)

    def add_connections(self, n_arr: list):
        for n in n_arr:
            self.add_connection(n)

    def receive_spike(self, input):
        self.input_spikes += torch.as_tensor(input, dtype=self.input_spikes.dtype)

    def send_spike(self):
        for i, c in enumerate(self.connections):
            c.receive_spike(self.spike * min(self.weights[i], 0.3))

    def time_step(self):
        self.spike, self.mem = self.neuron(self.input_spikes, self.mem)
        self.history.append({
            "input" : self.input_spikes,
            "spike" : self.spike,
            "mem": self.mem
        })
        self.input_spikes = torch.tensor(0.0)
            
class LSM():
    class Class_Neuron(Neuron):
        def __init__(self, beta = 0.9):
            super().__init__(beta)


    def __init__(self, size):
        self.size = size
        self.neurons = self.init_neurons(self.size)
        self.make_connections(self.neurons, num_connections = 2)
        self.connected_layers = []

    def init_neurons(self, num:int) -> list:
        self.neurons = []
        for i in range(num):
            self.neurons.append(self.Class_Neuron())
        
        return self.neurons
    
    def make_connections(self, neurons: list, num_connections = 1) -> list:
        for neuron in neurons:
            neuron.add_connections(random.sample(neurons, num_connections))

    def time_step(self, input, time_steps = 1):
        for _ in range(time_steps):
            for i, n in enumerate(self.neurons):
                n.receive_spike(input)
                n.time_step()
                n.send_spike()

    def readout(self) -> list:
        return [ n.mem for n in self.neurons ]

    def add_layer(self, l, weight):
        self.connected_layers.append( (l, weight) )

    def reset_state(self):
        for neuron in self.neurons:
            neuron.history = []
            neuron.input_spikes = torch.tensor(0.0)
            neuron.spike = torch.tensor(0.0)
            neuron.mem = torch.tensor(0.0)


def tokenize(text: str) -> list[str]:
    cleaned = text.lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ")
    return [token for token in cleaned.split() if token]


def build_dataset() -> list[tuple[str, int]]:
    positive = [
        "I loved this movie and the performances were excellent",
        "The film was inspiring warm and beautifully made",
        "A delightful story with charming characters",
        "This was a fantastic and uplifting experience",
        "The acting was brilliant and the pacing was perfect",
        "I enjoyed every minute of this great film",
        "What a thoughtful and genuinely moving story",
        "The direction was sharp and the ending felt satisfying",
    ]

    negative = [
        "I hated this movie and the performances were terrible",
        "The film was dull cold and badly made",
        "A frustrating story with annoying characters",
        "This was a boring and exhausting experience",
        "The acting was weak and the pacing was awful",
        "I disliked every minute of this bad film",
        "What a messy and completely unconvincing story",
        "The direction was sloppy and the ending felt empty",
    ]

    dataset = [(sentence, 1) for sentence in positive] + [(sentence, 0) for sentence in negative]
    random.shuffle(dataset)
    return dataset


def train_fasttext(dataset: list[tuple[str, int]]) -> FastText:
    corpus = [tokenize(sentence) for sentence, _ in dataset]
    model = FastText(
        vector_size=8,
        window=3,
        min_count=1,
        workers=1,
        sg=1,
        seed=42,
    )
    model.build_vocab(corpus)
    model.train(corpus, total_examples=len(corpus), epochs=30)
    return model


def sentence_to_input_sequence(model: FastText, sentence: str, num_steps: int = 20) -> torch.Tensor:
    tokens = tokenize(sentence)
    if not tokens:
        return torch.zeros(num_steps, dtype=torch.float32)

    word_vectors = np.abs(np.array([model.wv[token] for token in tokens], dtype=np.float32))
    spikes = spikegen.latency(torch.from_numpy(word_vectors), num_steps=num_steps, normalize=True)
    return spikes.float().mean(dim=(1, 2))


def encode_sentence(reservoir: LSM, model: FastText, sentence: str, num_steps: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    reservoir.reset_state()
    input_sequence = sentence_to_input_sequence(model, sentence, num_steps=num_steps)

    for step_value in input_sequence:
        reservoir.time_step(step_value, time_steps=1)

    reservoir_state = torch.stack([neuron.mem for neuron in reservoir.neurons]).float()
    spike_history = torch.stack([
        neuron.history[-1]["spike"].detach().float() if neuron.history else torch.tensor(0.0)
        for neuron in reservoir.neurons
    ])
    return reservoir_state, spike_history


def build_features(reservoir: LSM, model: FastText, samples: list[tuple[str, int]], num_steps: int = 20):
    features = []
    labels = []
    traces = []

    with torch.no_grad():
        for sentence, label in samples:
            reservoir_state, spike_snapshot = encode_sentence(reservoir, model, sentence, num_steps=num_steps)
            features.append(reservoir_state)
            labels.append(label)
            traces.append((sentence, label, reservoir_state, spike_snapshot))

    return torch.stack(features), torch.tensor(labels, dtype=torch.float32), traces


def standardize(train_features: torch.Tensor, test_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_features - mean) / std, (test_features - mean) / std


def train_readout(train_features: torch.Tensor, train_labels: torch.Tensor) -> nn.Module:
    classifier = nn.Linear(train_features.shape[1], 1)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(200):
        optimizer.zero_grad()
        logits = classifier(train_features).squeeze(-1)
        loss = loss_fn(logits, train_labels)
        loss.backward()
        optimizer.step()

    return classifier


def evaluate(classifier: nn.Module, features: torch.Tensor, labels: torch.Tensor):
    with torch.no_grad():
        logits = classifier(features).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()
        accuracy = (predictions == labels).float().mean().item()
    return accuracy, probabilities, predictions


def confusion_matrix(labels: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros((2, 2), dtype=torch.int64)
    for true_label, predicted_label in zip(labels.int(), predictions.int()):
        matrix[true_label, predicted_label] += 1
    return matrix


def analyze_reservoir(traces, classifier: nn.Module):
    positive_states = [state for _, label, state, _ in traces if label == 1]
    negative_states = [state for _, label, state, _ in traces if label == 0]

    positive_mean = torch.stack(positive_states).mean(dim=0)
    negative_mean = torch.stack(negative_states).mean(dim=0)
    separation = positive_mean - negative_mean
    top_neurons = torch.topk(torch.abs(separation), k=min(5, separation.numel()))

    weights = classifier.weight.detach().squeeze(0)
    top_weight_neurons = torch.topk(torch.abs(weights), k=min(5, weights.numel()))

    print("\nLSM analysis")
    print(f"  positive mean membrane potential: {positive_mean.mean().item():.4f}")
    print(f"  negative mean membrane potential: {negative_mean.mean().item():.4f}")
    print(f"  mean absolute class separation: {torch.abs(separation).mean().item():.4f}")
    print("  most class-sensitive neurons:")
    for index in top_neurons.indices.tolist():
        print(
            f"    neuron {index:02d}: positive={positive_mean[index].item():.4f} "
            f"negative={negative_mean[index].item():.4f} diff={separation[index].item():+.4f}"
        )

    print("  strongest readout weights:")
    for index in top_weight_neurons.indices.tolist():
        print(f"    neuron {index:02d}: weight={weights[index].item():+.4f}")


def main():
    dataset = build_dataset()
    train_samples = dataset[:12]
    test_samples = dataset[12:]

    fasttext = train_fasttext(dataset)
    reservoir = LSM(5000)

    train_features, train_labels, train_traces = build_features(reservoir, fasttext, train_samples)
    test_features, test_labels, test_traces = build_features(reservoir, fasttext, test_samples)

    train_features, test_features = standardize(train_features, test_features)

    classifier = train_readout(train_features, train_labels)

    train_accuracy, train_probabilities, train_predictions = evaluate(classifier, train_features, train_labels)
    test_accuracy, test_probabilities, test_predictions = evaluate(classifier, test_features, test_labels)

    print("FastText + LSM sentiment analysis")
    print(f"  training samples: {len(train_samples)}")
    print(f"  test samples: {len(test_samples)}")
    print(f"  train accuracy: {train_accuracy:.3f}")
    print(f"  test accuracy: {test_accuracy:.3f}")

    print("\nTraining confusion matrix [true rows x predicted cols]")
    print(confusion_matrix(train_labels, train_predictions))
    print("\nTest confusion matrix [true rows x predicted cols]")
    print(confusion_matrix(test_labels, test_predictions))

    analyze_reservoir(train_traces, classifier)

    print("\nPrediction analysis on test samples")
    for (sentence, label), probability, prediction, trace in zip(test_samples, test_probabilities, test_predictions, test_traces):
        true_name = "positive" if label == 1 else "negative"
        predicted_name = "positive" if int(prediction.item()) == 1 else "negative"
        mean_spike = trace[3].mean().item()
        print(
            f"  [{true_name:8s} -> {predicted_name:8s}] prob={probability.item():.3f} "
            f"mean_spike={mean_spike:.4f} | {sentence}"
        )

    misclassified = [
        (sentence, label, probability.item(), int(prediction.item()))
        for (sentence, label), probability, prediction in zip(test_samples, test_probabilities, test_predictions)
        if int(prediction.item()) != label
    ]
    print("\nPrediction summary")
    print(f"  correct predictions: {len(test_samples) - len(misclassified)} / {len(test_samples)}")
    print(f"  misclassified samples: {len(misclassified)}")


if __name__ == "__main__":
    main()
