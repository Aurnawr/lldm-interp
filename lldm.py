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
V =len(CHARS) + 1
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

