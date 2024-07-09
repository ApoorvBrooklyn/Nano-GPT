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
val_data = data[n:]


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
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

xb, yb = get_batch('train')
print('inputs: ')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)


print("-----------------------")

for b in range(batch_size): # Batch Dimension
    for t in range(block_size): #time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"When input is {context.tolist()} the target is: {target}")


import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embeddings_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embeddings_table(idx) # Batch, time, Channel tensor
        # Here Batch is 4, time is 8, channel is vocab_size = 65
        # loss = F.cross_entropy(logits, targets)
        # Above is actual method but in doc we will find that F.cross expects Channel as Second input here we have it as third 
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the prediction
            logits, loss = self(idx)
            ## Focus only on last step/layer
            logits = logits[:,-1,:] # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples =1) # (B,1)
            # append idx_next to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


    
# Technically Logits are the predictions and BigramLAnguageModel is just simple Neural network with Forward pass    
m = BigramLanguageModel(vocab_Size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
# Loss here is 4.87 which can be estimated by -ln(1/65) ie -ln(1/vocab_size) which is around 4.17

## We are creating a tensor of (1,1) dimension
# idx = torch.zeros((1,1), dtype = torch.long)
## Since the generate works on batches we have to unpluck single batchwise elements which gives us Time Steps which are converted into List using tolist() function
print(decode(m.generate(idx = torch.zeros((1,1), dtype = torch.long), max_new_tokens=100)[0].tolist()))  


optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(5000):
    xb,yb = get_batch('train')

    # Evaluate Loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(loss.item())

print(decode(m.generate(idx = torch.zeros((1,1), dtype = torch.long), max_new_tokens=400)[0].tolist()))  

