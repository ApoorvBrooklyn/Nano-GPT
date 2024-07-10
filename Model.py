import torch
import torch
import torch.nn as nn
from torch.nn import functional as F


# Hyper Parameter 
batch_size = 32
block_size = 64
max_iters = 5000
learning_rate= 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
eval_interval = 500

torch.manual_seed(1337)


# Reading the txt

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()



chars = sorted(list(set(text)))
vocab_size = len(chars)
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


#block_size = 8
# x = train_data[:block_size]
# y = train_data[1:block_size+1]

# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     # print(f"When input is {context} the target is {target}")



# batch_size = 4 # Number of independent sequences we will generate
# block_size = 8 # What is maximum context length for prediction
# Data Loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

# xb, yb = get_batch('train')
# print('inputs: ')
# print(xb.shape)
# print(xb)
# print('targets:')
# print(yb.shape)
# print(yb)


# print("-----------------------")

# for b in range(batch_size): # Batch Dimension
#     for t in range(block_size): #time dimension
#         context = xb[b, :t+1]
#         target = yb[b,t]
#         print(f"When input is {context.tolist()} the target is: {target}")



# torch.manual_seed(1337)
@torch.no_grad() 
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """"Attention Vectors"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C **-0.5 # (B,T,C) @ (B,T,C) ----> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) # Masking the upper half of triangle so it doesnt communicate with past
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v # (B,T,T) @ (B, T, C) -----> (B,T,C)
        return out

class MultiheadAttention(nn.Module):
    """Multiple heads of self-attention in parallels for better communication"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for __ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        # self.proj = nn.Linear(head_seize * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """Simple Feed Forward Network"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout), )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """Transformer Block: Communication followed by computation"""
    def __init__(self, n_embd, n_head):
        ## n_embd : Embeddings dimension, n_nead: number of heads we would like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiheadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

# Simple Bigram Model
class BigramLanguageModel(nn.Module):


    def __init__(self): # We can add vocab_size here !! Wait for time being
        super().__init__()
        #each token directly reads off the logits for the next token from Lookup table
        self.token_embeddings_table = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            nn.LayerNorm(n_layer)
        )
        self.ln_f = nn.LayerNorm(n_embd) # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # self.sa_head = MultiheadAttention(4,n_embd//4) ## i.e.  heads of 8 dimensional self-attention vectors
        # self.ffwd = FeedForward(n_embd) Already included in Blocks block

        self.apply(self._init_weights) # not in Tutorial

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        

    def forward(self, idx, targets=None):
        B,T = idx.shape

        # idx and targets are both (B,T) tensor of integer
        tok_emb = self.token_embeddings_table(idx)
        pos_emb = self.position_embeddings_table(torch.arange(T, device = device)) # T,C
        x = tok_emb + pos_emb #(B,T,C)
        # x = self.sa_head(x) ## Apply one head of self attention
        # x = self.ffwd(x)

        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T)
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
            # crop idx to last block_size_token
            idx_cond = idx[:,-block_size:]
            # get the prediction
            logits, loss = self(idx_cond)
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
model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
# logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)
# Loss here is 4.87 which can be estimated by -ln(1/65) ie -ln(1/vocab_size) which is around 4.17

## We are creating a tensor of (1,1) dimension
# idx = torch.zeros((1,1), dtype = torch.long)
## Since the generate works on batches we have to unpluck single batchwise elements which gives us Time Steps which are converted into List using tolist() function
# print(decode(m.generate(idx = torch.zeros((1,1), dtype = torch.long), max_new_tokens=100)[0].tolist()))  


optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # For every fe interval we will evaluate model
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss{losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample of data
    xb, yb = get_batch('train')

    # Evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# for steps in range(5000):
#     xb,yb = get_batch('train')

#     # Evaluate Loss
#     logits, loss = m(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

#     print(loss.item())
context = torch.zeros((1,1), dtype=torch.long, device = device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

