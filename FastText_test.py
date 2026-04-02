from gensim.models import FastText
import snntorch as snn
from snntorch import spikegen
import torch
import numpy as np

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