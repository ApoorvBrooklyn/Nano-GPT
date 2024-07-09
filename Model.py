import torch

# Reading the txt

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(len(text))


chars = sorted(list(set(text)))
vocab_Size = len(chars)
# print(''.join(chars))
# print(vocab_Size)

## Creating Mapping from characters to integers

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype = torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])


# Splitting the data into training and validation

n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]


block_size = 8
x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    # print(f"When input is {context} the target is {target}")


torch.manual_seed(1337)
batch_size = 4 # Number of independent sequences we will generate
block_size = 8 # What is maximum context length for prediction

def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i:i+block_size+1] for i in ix])
    return x,y

xb, yb = get_batch('train')
print('inputs: ')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)


print("-----------------------")

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"When input is {context} the target is {target}")


import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embeddings_table = nn.Embedding(vocab_size, vocab_size)
        

