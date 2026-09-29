import random, torch, torch.nn as nn, torch.nn.functional as F

K = 4
CHARS = list("0123456789+=>")
STOI = {c: i for i, c in enumerate(CHARS)}
MASK_ID = len(CHARS)                 # the absorbing state
V = len(CHARS) + 1
PROMPT_LEN, RESP_LEN = 2 * K, K + 1
SEQ_LEN = PROMPT_LEN + RESP_LEN
RESP_LABELS = [f"r{i}" for i in range(1, K)] + [">", "ANS"]

def make_example():
    ops = [random.randint(0, 9) for _ in range(K)]
    s, partial = ops[0], []
    for o in ops[1:]:
        s = (s + o) % 10
        partial.append(str(s))
    text = "+".join(map(str, ops)) + "=" + "".join(partial) + ">" + str(s)
    return [STOI[c] for c in text]

def get_batch(B, device):
    return torch.tensor([make_example() for _ in range(B)], device=device)

class MaskPredictor(nn.Module):
    def __init__(self, d=128, n_layers=4, n_heads=4):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(SEQ_LEN, d)
        layer = nn.TransformerEncoderLayer(
            d, n_heads, 4 * d, dropout=0.0, batch_first=True,
            norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)

    def forward(self, x):                      # x: (B, L) token ids, some = MASK_ID
        h = self.tok(x) + self.pos(torch.arange(x.size(1), device=x.device))
        return self.head(self.norm(self.enc(h)))   # (B, L, V) logits

def llada_loss(model, x, prompt_len=PROMPT_LEN, eps=1e-3, sft=True):
    B, L = x.shape
    t = (1 - eps) * torch.rand(B, 1, device=x.device) + eps     # avoid t = 0
    maskable = torch.ones_like(x, dtype=torch.bool)
    if sft:
        maskable[:, :prompt_len] = False
    masked = (torch.rand(B, L, device=x.device) < t) & maskable
    xt = torch.where(masked, torch.full_like(x, MASK_ID), x)
    ce = F.cross_entropy(model(xt).transpose(1, 2), x, reduction="none")  # (B, L)
    return ((ce * masked) / t).sum(1).div(maskable.sum(1)).mean()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MaskPredictor().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
for it in range(3000):
    loss = llada_loss(model, get_batch(256, device))
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 500 == 0:
        print(it, loss.item())

@torch.no_grad()
def generate(model, prompt, gen_len=RESP_LEN, steps=RESP_LEN, block_len=None,
             temperature=0.0, remasking="low_confidence"):
    B, P = prompt.shape
    block_len = block_len or gen_len
    n_blocks = gen_len // block_len
    spb = steps // n_blocks                       # steps per block
    x = torch.cat([prompt, torch.full((B, gen_len), MASK_ID, device=prompt.device)], 1)
    commit_step = torch.full((B, gen_len), -1, dtype=torch.long)
    pred_hist, step = [], 0

    for b in range(n_blocks):
        lo, hi = P + b * block_len, P + (b + 1) * block_len
        n = (x[:, lo:hi] == MASK_ID).sum(1)       # linear schedule over this block
        sched = (n // spb)[:, None].repeat(1, spb)
        sched += (torch.arange(spb, device=x.device)[None] < (n % spb)[:, None]).long()

        for s in range(spb):
            is_mask = x == MASK_ID
            logits = model(x)
            if temperature > 0:
                g = -torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-20)))
                x0 = (logits / temperature + g).argmax(-1)
            else:
                x0 = logits.argmax(-1)
            pred_hist.append(torch.where(is_mask, x0, x)[:, P:].cpu())  # current belief

            if remasking == "low_confidence":
                conf = F.softmax(logits, -1).gather(-1, x0[..., None]).squeeze(-1)
            else:                                                   # "random"
                conf = torch.rand(x0.shape, device=x.device)
            conf = conf.masked_fill(~is_mask, -float("inf"))
            conf[:, hi:] = -float("inf")                           # current block only

            for i in range(B):
                k = int(sched[i, s])
                if k == 0:
                    continue
                idx = conf[i].topk(k).indices
                x[i, idx] = x0[i, idx]                             # irreversible
                commit_step[i, (idx - P).cpu()] = step
            step += 1
    return x, commit_step, torch.stack(pred_hist, 1)   # hist: (B, steps, gen_len)

def stabilization_step(hist, final):
    B, S, L = hist.shape
    same = (hist == final[:, None, :]).int()
    stable_from = torch.flip(torch.cumprod(torch.flip(same, [1]), 1), [1]).bool()
    first = stable_from.float().argmax(1).float()
    return torch.where(stable_from.any(1), first, torch.full_like(first, float(S)))

model.eval()
data = get_batch(1000, device)
for remasking in ["low_confidence", "random"]:
    out, commit, hist = generate(model, data[:, :PROMPT_LEN], remasking=remasking)
    resp = out[:, PROMPT_LEN:]
    stable = stabilization_step(hist, resp.cpu())
    acc = (resp[:, -1] == data[:, -1]).float().mean().item()
    print(f"{remasking}: answer acc {acc:.3f}")
    for j, name in enumerate(RESP_LABELS):
        print(f"  {name:>4}  commit {commit[:, j].float().mean():.2f}"
              f"  stable {stable[:, j].mean():.2f}")