import argparse, csv, json, os, random, torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--K", type=int, default=10)
p.add_argument("--answer", choices=["last", "mod7"], default="last")  # "last" = shortcut (ANS copies r_{K-1})
p.add_argument("--iters", type=int, default=20000)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--n_eval", type=int, default=1000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--ckpt", default=None, help="load this checkpoint and skip training")
p.add_argument("--out", default="results", help="each run writes to <out>/<run name>/")
args = p.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu") if args.ckpt else None
if ckpt:                                  # task shape must match the trained model
    args.K, args.answer = ckpt["config"]["K"], ckpt["config"]["answer"]

random.seed(args.seed); torch.manual_seed(args.seed)

K = args.K
CHARS = list("0123456789+=>")
STOI = {c: i for i, c in enumerate(CHARS)}
MASK_ID = len(CHARS)                 # the absorbing state
V = len(CHARS) + 1
PROMPT_LEN, RESP_LEN = 2 * K, K + 1
SEQ_LEN = PROMPT_LEN + RESP_LEN
RESP_LABELS = [f"r{i}" for i in range(1, K)] + [">", "ANS"]

ANSWER_RULES = {
    "last": lambda ops, partial: partial[-1],          # copyable from the reasoning
    "mod7": lambda ops, partial: sum(ops) % 7,         # not copyable from any r
}

def make_example(answer=args.answer):
    ops = [random.randint(0, 9) for _ in range(K)]
    s, partial = ops[0], []
    for o in ops[1:]:
        s = (s + o) % 10
        partial.append(s)
    ans = ANSWER_RULES[answer](ops, partial)
    text = "+".join(map(str, ops)) + "=" + "".join(map(str, partial)) + ">" + str(ans)
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


run_name = f"K{K}-{args.answer}-s{args.seed}" + ("-eval" if ckpt else "")
run_dir = os.path.join(args.out, run_name)
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "config.json"), "w") as f:
    json.dump(vars(args), f, indent=2)

# training loop

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MaskPredictor().to(device)
if ckpt:
    model.load_state_dict(ckpt["model"])
else:
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    loss_log = []
    for it in range(args.iters):
        loss = llada_loss(model, get_batch(args.batch, device))
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 50 == 0:
            loss_log.append((it, loss.item()))
        if it % 500 == 0:
            print(it, loss.item())
    with open(os.path.join(run_dir, "train_loss.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("iter", "loss")] + loss_log)
    os.makedirs("checkpoints", exist_ok=True)
    path = f"checkpoints/{run_name}.pt"
    torch.save({"model": model.state_dict(), "config": vars(args)}, path)
    print("saved", path)

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
random.seed(args.seed + 1); torch.manual_seed(args.seed + 1)   # same eval whether or not we trained
data = get_batch(args.n_eval, device)
target = data[:, PROMPT_LEN:].cpu()
rows, summary = [("remasking", "position", "acc", "commit", "stable")], {}
for remasking in ["low_confidence", "random"]:
    out, commit, hist = generate(model, data[:, :PROMPT_LEN], remasking=remasking)
    resp = out[:, PROMPT_LEN:].cpu()
    stable = stabilization_step(hist, resp)
    pos_acc = (resp == target).float().mean(0)
    metrics = {f"{remasking}/answer_acc": pos_acc[-1].item(),
               f"{remasking}/reasoning_acc": (resp[:, :K - 1] == target[:, :K - 1]).all(1).float().mean().item()}
    print(f"{remasking}: answer acc {pos_acc[-1]:.3f}")

    # Does the written ANS agree with the written reasoning? Only defined when the
    # answer is a function of the partial sums, i.e. the "last" rule.
    if args.answer == "last":
        follows = resp[:, -1] == resp[:, K - 2]
        wrong_r = resp[:, K - 2] != target[:, K - 2]
        metrics[f"{remasking}/ans_follows_reasoning"] = follows.float().mean().item()
        metrics[f"{remasking}/n_wrong_last_r"] = wrong_r.sum().item()
        if wrong_r.any():             # when the reasoning is wrong: follow it, or the truth?
            metrics[f"{remasking}/follows_given_wrong_r"] = follows[wrong_r].float().mean().item()
            metrics[f"{remasking}/correct_given_wrong_r"] = (resp[wrong_r, -1] == target[wrong_r, -1]).float().mean().item()
        print(f"  ANS follows written r{K - 1}: {follows.float().mean():.3f}")

    for j, name in enumerate(RESP_LABELS):
        c, st = commit[:, j].float().mean().item(), stable[:, j].mean().item()
        rows.append((remasking, name, pos_acc[j].item(), c, st))
        metrics.update({f"{remasking}/acc/{name}": pos_acc[j].item(),
                        f"{remasking}/commit/{name}": c, f"{remasking}/stable/{name}": st})
        print(f"  {name:>4}  acc {pos_acc[j]:.3f}  commit {c:.2f}  stable {st:.2f}")
    summary.update(metrics)

with open(os.path.join(run_dir, "per_position.csv"), "w", newline="") as f:
    csv.writer(f).writerows(rows)
with open(os.path.join(run_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("results in", run_dir)
