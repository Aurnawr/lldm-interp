import torch 
import random
import torch.nn as nn
import torch.nn.functional as F

"""Task - The task is a chain of single-digit additions mod 10, where the response writes each partial
sum and then the answer. Note the built-in shortcut: the answer equals the last partial sum r3"""

k = 4

CHARS = list("123456789+=>")
STOI = {c : i for i, c in enumerate (CHARS)}
MASK_ID = len(CHARS) # the absorbing state
V =len(CHARS) + 1 # the discrete integer indices 
PROMPT_LEN, RESP_LEN = 2*k, k + 1
SEQ_LEN = PROMPT_LEN + RESP_LEN
RESP_LABELS = [f"r{i}" for i in range(1, K)] + [">", "ANS"]

def make_example(): # function to make training examples
    ops = [random.randint(0,9) for _ in range(k)]
    s, partial = ops[0], []
    for o in ops[1:]:
        s = (s + 0) % 10
        partial.append(str(s))
        text = "+".join(map(str,ops)) + "="+ "".join(partial) + ">" + str(s)
        return [STOI[c] for c in text]

def get_batch(B, device):
    return torch.tensor([make_example() for _ in range (B)], device = device)

# the mask predictor

class MaskPredictor(nn.Module): 
    def __init__(self,d = 128, n_heads = 4, n_layers = 4): # whenever this class object is called, all these layers get initialized
        super().__init__()
        self.tok = nn.Embedding(V,d)
        self.pos = nn.Embedding(SEQ_LEN,d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads,dim_feedforward=4*d, batch_first=True, dropout = 0.0, norm_first=True, activation='gelu' )
        self.enc = nn.TransformerEncoder(layer,n_layers, enable_nested_tensor=False)  # using an encoder(the one change from autoregressive models)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d,V)

    def forward(self, x):  # x: (B,L) token_ids
        h = self.tok(x) + self.pos(torch.arange(x.size(1),device=x.device))
        return self.head(self.norm(self.enc(h)))  # (B,L,V) logits

# defining the loss
def llada_loss(model,x,prompt_len = PROMPT_LEN,eps =1e-3, sft = True):
    B,L = x.shape
    t = (1 - eps) * torch.rand(B, 1, device=x.device) + eps # each of the batch gets a single t between 0-1 
    maskable = torch.ones_like(x,dtype=torch.bool)
    if sft:
        maskable[:,:prompt_len] = False # cannot mask the prompt 
        masked = (torch.rand(B,L,device=x.device) < t) & maskable
        xt = torch.where(masked, torch.full_like(x, MASK_ID), x) 
        ce = F.cross_entropy(model(xt).transpose(1, 2), x, reduction="none") # (B,L)
        return ((ce * masked) / t).sum(1).div(maskable.sum(1)).mean()