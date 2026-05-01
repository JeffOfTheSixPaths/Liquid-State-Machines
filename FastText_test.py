from gensim.models import FastText # need to cite this paper
import snntorch as snn
from snntorch import spikegen
import torch
import numpy as np


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
        self.input_spikes += torch.tensor(input)

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
        self.neurons = self.init_neurons(self.size)
        self.make_connections(self.neurons, num_connections = 2)
        self.connected_layers = []

    def init_neurons(self, num:int) -> list:
        self.neurons = []
        for i in range(num):
            self.neurons.append(Class_Neuron())
        
        return neurons
    
    def make_connections(neurons: list, num_connections = 1) -> list:
        for neuron in neurons:
            neuron.add_connections(random.sample(neurons, num_connections))

    def time_step(input, time_steps = 1):
        for _ in range(time_steps):
            for i, n in enumerate(self.neurons):
                n.receive_spike(input)
                n.time_step()
                n.send_spike()

    def readout() -> list:
        return [ n.mem for n in self.neurons ]

    def add_layer(l, weight):
        self.connected_layers.append( (l, weight) )


num_classes = 2
# Sample data
sentences = [
    ['this',"is", 'a', 'sample', 'document']
]

# Initialize the FastText model
model = FastText(sentences, vector_size=5, window=5, min_count=1, workers=4)

# Train the model
model.train(sentences, total_examples=len(sentences), epochs=10)

# Get vector for a word
print(model.wv)

for i in model.wv.key_to_index:
    print(i, model.wv[i])


# need to convert the word vectors to a spike train
#going to use latency encodings

data = np.abs(np.array([model.wv[i] for i in model.wv.key_to_index]) )
spikes = spikegen.latency(torch.from_numpy(data), num_steps = 20, normalize = True)

# for i in data:
#     print(spikegen.latency(torch.from_numpy(i), num_steps = 20, normalize = True))

print(spikes)
print(len(spikes))

l = LSM(20)
