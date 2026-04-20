import numpy as np
import pandas as pd
import torch 
import snntorch as snn

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

class Layer(nn.Module): # just an SNN
    def __init__(self, input_size, output_size):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(input_size, 1000) # needs to take the size from the previous layer
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(self.output_size, 10)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.connected_layers = []

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
    
    def add_layer(l, weight):
        self.connected_layers.append( (l, weight) )

class Answer():
    pass # this should be either a feedforward network or a regression layer

#simulation
df = 0 # "dataframe"
time_steps = 100 # this is defined from the previous language
for _ in time_steps:
    sample = 1 # sample from df


    pass
