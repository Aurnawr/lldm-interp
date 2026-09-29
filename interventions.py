"""Causal checks on a trained checkpoint: what does ANS depend on?

For each condition the response is partly masked and/or corrupted, ANS is always
masked, and we read the model's ANS argmax from a single forward pass.
Read-only on checkpoints; writes results/<ckpt name>/interventions.csv.

    python interventions.py --ckpt checkpoints/K8-mod7-s0.pt
"""

"""It works for any K and either answer rule. For each checkpoint it prints ANS accuracy under several conditions:

the whole response masked (step 0)
the true reasoning visible
all reasoning replaced with random digits
each partial sum corrupted on its own, set to a digit guaranteed to be wrong
"""

import argparse, csv, os, random, sys, torch

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--n", type=int, default=4000)
p.add_argument("--seed", type=int, default=123)
p.add_argument("--out", default="results")
args = p.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu")
cfg = ckpt["config"]
sys.argv = ["lldm.py", "--K", str(cfg["K"]), "--answer", cfg["answer"]]   # lldm sets task shape at import
import lldm as L

device = "cuda" if torch.cuda.is_available() else "cpu"
model = L.MaskPredictor().to(device)
model.load_state_dict(ckpt["model"]); model.eval()

random.seed(args.seed); torch.manual_seed(args.seed)
data = L.get_batch(args.n, device)
K, P, A = L.K, L.PROMPT_LEN, L.SEQ_LEN - 1
R = list(range(P, P + K - 1))                  # reasoning positions r1..r_{K-1}
target = data[:, A]

@torch.no_grad()
def predict(x):                                 # x: (B, L) -> argmax at every response position
    return model(x)[:, P:].argmax(-1)

def wrong_digits(col):                          # a digit guaranteed to differ from the true one
    return (col + torch.randint(1, 10, col.shape, device=device)) % 10

conditions = {}
x = data.clone(); x[:, P:] = L.MASK_ID
conditions["all masked (step 0)"] = x
x = data.clone(); x[:, A] = L.MASK_ID
conditions["true reasoning"] = x
x = data.clone(); x[:, R] = torch.randint(0, 10, (args.n, K - 1), device=device); x[:, A] = L.MASK_ID
conditions["random reasoning"] = x
for i, pos in enumerate(R):
    x = data.clone(); x[:, pos] = wrong_digits(x[:, pos]); x[:, A] = L.MASK_ID
    conditions[f"corrupt r{i + 1} only"] = x

base = predict(conditions["true reasoning"])[:, -1]
rows = [("condition", "ans_acc", "ans_changed")]
print(f"{args.ckpt}  (K={K}, answer={cfg['answer']}, n={args.n})")
print(f"  {'condition':<22} ans_acc  changed")
for name, x in conditions.items():
    ans = predict(x)[:, -1]
    acc = (ans == target).float().mean().item()
    changed = (ans != base).float().mean().item()   # vs. prediction given the true reasoning
    rows.append((name, acc, changed))
    print(f"  {name:<22} {acc:.3f}    {changed:.3f}")

# Which reasoning tokens can the model already produce from the prompt alone?
step0 = predict(conditions["all masked (step 0)"])
acc0 = (step0 == data[:, P:]).float().mean(0)
print("  step-0 acc per position:", "  ".join(f"{n} {a:.2f}" for n, a in zip(L.RESP_LABELS, acc0.tolist())))
rows += [("step0 acc " + n, a, "") for n, a in zip(L.RESP_LABELS, acc0.tolist())]

run_dir = os.path.join(args.out, os.path.splitext(os.path.basename(args.ckpt))[0])
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "interventions.csv"), "w", newline="") as f:
    csv.writer(f).writerows(rows)
print("wrote", os.path.join(run_dir, "interventions.csv"))
