import numpy as np
import pandas as pd
import torch 
import snntorch as snn

            
class LSM():
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


    def __init__(self, size):
        neurons = init_neurons(self.size)
        make_connections(neurons, num_connections = 2)

    def init_neurons(self, num:int) -> list:
        neurons = []
        for i in range(num):
            neurons.append(Neuron())
        
        return neurons
    
    def make_connections(neurons: list, num_connections = 1) -> list:
        for neuron in neurons:
            neuron.add_connections(random.sample(neurons, num_connections))

    def simulate(time_steps : int):
        for _ in range(time_steps):
            for i, n in enumerate(neurons):
                n.receive_spike(min(np.random.random(), 0.4))
                n.time_step()
                n.send_spike()


        