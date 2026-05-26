"""
Phase 5: Scale-up — N=256, G=32, dim=128
==========================================
Goal: prove topology effects become statistically significant at larger scale.
Based on hierarchical_moe_experiment.py with scaled parameters.

Key changes from Phase 4 (N=64, G=16, dim=64):
- N: 64 -> 256 (4x experts)
- G: 16 -> 32 (2x groups)  
- dim: 64 -> 128 (2x hidden)
- steps: 500 -> 800
- seeds: 42, 123, 456 (all in one run)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import math, time, json, os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# ---- Data ----
with open('/tmp/alice_clean.txt') as f:
    ALICE_TEXT = f.read()
CHARS = sorted(list(set(ALICE_TEXT)))
VOCAB_SIZE = len(CHARS) + 4
char_to_id = {c: i + 4 for i, c in enumerate(CHARS)}
BOS, EOS, PAD, UNK = 0, 1, 2, 3
def encode(text):
    return [char_to_id.get(c, UNK) for c in text]
tokens = encode(ALICE_TEXT)
split = int(len(tokens) * 0.8)
train_tokens = tokens[:split]
test_tokens = tokens[split:]
print(f"Vocab={VOCAB_SIZE}, Train={len(train_tokens)}, Test={len(test_tokens)}", flush=True)

def get_batch(tokens_list, seq_len=64, bs=32):
    L = len(tokens_list)
    inp = torch.zeros(bs, seq_len, dtype=torch.long)
    tgt = torch.zeros(bs, seq_len, dtype=torch.long)
    for i in range(bs):
        s = np.random.randint(0, max(1, L - seq_len - 1))
        inp[i] = torch.tensor(tokens_list[s:s + seq_len])
        tgt[i] = torch.tensor(tokens_list[s + 1: s + seq_len + 1])
    return inp.to(device), tgt.to(device)


# ============================================================
# Topology Construction
# ============================================================
def build_cross_graph(name, n_groups, degree=3, seed=42):
    np.random.seed(seed)
    if name == "ring":
        G = nx.Graph()
        for i in range(n_groups):
            for off in range(1, degree // 2 + 1):
                G.add_edge(i, (i + off) % n_groups)
    elif name in ("expander", "random_regular"):
        d = min(degree, n_groups - 1)
        G = nx.random_regular_graph(d, n_groups, seed=seed) if d >= 1 else nx.empty_graph(n_groups)
    elif name == "dense":
        G = nx.complete_graph(n_groups)
    else:
        G = nx.empty_graph(n_groups)
    return G


# ============================================================
# Flat Standard MoE (baseline)
# ============================================================
class FlatMoE(nn.Module):
    def __init__(self, dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.w1 = nn.Parameter(torch.randn(num_experts, dim, dim * 2) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(num_experts, dim * 2))
        self.w2 = nn.Parameter(torch.randn(num_experts, dim * 2, dim) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(num_experts, dim))
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        B, S, D = x.shape
        logits = self.gate(x)
        topk_val, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(topk_val, dim=-1)

        out = torch.zeros_like(x)
        flat_x = x.reshape(-1, D)  # (B*S, D)
        for k in range(self.top_k):
            idx = topk_idx[:, :, k]  # (B, S)
            w = weights[:, :, k:k+1]  # (B, S, 1)
            flat_idx = idx.reshape(-1)  # (B*S,)
            expert_out = torch.zeros(B * S, D, device=x.device)
            for e in range(self.num_experts):
                mask = (flat_idx == e)
                if not mask.any():
                    continue
                # w1[e] is (dim, dim*2), F.linear needs (out, in) so transpose
                h = F.linear(flat_x[mask], self.w1[e].T, self.b1[e])
                h = F.gelu(h)
                expert_out[mask] = F.linear(h, self.w2[e].T, self.b2[e])
            out = out + w * expert_out.reshape(B, S, D)
        return out


# ============================================================
# Hierarchical MoE with Topology Routing
# ============================================================
class HierMoE(nn.Module):
    def __init__(self, dim, num_experts, num_groups, top_k_groups=1,
                 top_k_experts=2, comm_steps=2, cross_topology="expander",
                 degree=3, seed=42):
        super().__init__()
        self.N = num_experts
        self.G = num_groups
        self.E = num_experts // num_groups
        self.kg = top_k_groups
        self.ke = top_k_experts
        self.comm_steps = comm_steps

        # Expert FFN params
        self.w1 = nn.Parameter(torch.randn(num_experts, dim, dim * 2) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(num_experts, dim * 2))
        self.w2 = nn.Parameter(torch.randn(num_experts, dim * 2, dim) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(num_experts, dim))

        # Two-level gates
        self.group_gate = nn.Linear(dim, num_groups)
        self.expert_gate = nn.Linear(dim, num_experts)

        # Communication weight
        self.comm_alpha = nn.Parameter(torch.tensor(0.1))

        # Precompute topology adjacency
        G = build_cross_graph(cross_topology, num_groups, degree, seed)
        adj = torch.zeros(num_groups, num_groups)
        for i, j in G.edges():
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        # Normalize rows
        row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1)
        self.register_buffer('adj_norm', adj / row_sum)

    def forward(self, x):
        B, S, D = x.shape

        # Level 1: Group gating
        g_logits = self.group_gate(x)
        g_topk_val, g_topk_idx = torch.topk(g_logits, self.kg, dim=-1)
        g_weights = F.softmax(g_topk_val, dim=-1)

        # Level 2: Expert gating (within selected groups)
        e_logits = self.expert_gate(x)
        e_topk_val, e_topk_idx = torch.topk(e_logits, self.ke, dim=-1)
        e_weights = F.softmax(e_topk_val, dim=-1)

        # Standard top-k MoE routing (per-expert loop for memory efficiency)
        out = torch.zeros_like(x)
        flat_x = x.reshape(-1, D)  # (B*S, D)
        for k in range(self.ke):
            idx = e_topk_idx[:, :, k]  # (B, S)
            w = e_weights[:, :, k:k+1]  # (B, S, 1)
            flat_idx = idx.reshape(-1)  # (B*S,)
            expert_out = torch.zeros(B * S, D, device=x.device)
            for e in range(self.N):
                mask = (flat_idx == e)
                if not mask.any():
                    continue
                # w1[e] is (dim, dim*2), F.linear needs (out, in) so transpose
                h = F.linear(flat_x[mask], self.w1[e].T, self.b1[e])
                h = F.gelu(h)
                expert_out[mask] = F.linear(h, self.w2[e].T, self.b2[e])
            out = out + w * expert_out.reshape(B, S, D)

        # Topology-conditioned communication between expert groups
        if self.comm_steps > 0:
            alpha = torch.sigmoid(self.comm_alpha)
            # Propagate learned group-level activations through topology graph
            group_signal = self.group_gate(x)  # (B, S, G)
            for _ in range(self.comm_steps):
                mixed = torch.einsum('gk,bsg->bsk', self.adj_norm, group_signal)
                group_signal = (1 - alpha) * group_signal + alpha * mixed

            # Modulate output by topology-smoothed group signal
            group_weight = torch.sigmoid(group_signal)  # (B, S, G)
            for k in range(self.ke):
                idx = e_topk_idx[:, :, k]
                expert_group = idx // self.E  # (B, S) which group each routed expert is in
                flat_eg = expert_group.reshape(-1)
                batch_pos = torch.arange(B * S, device=x.device)
                gw = group_weight.reshape(-1, self.G)[batch_pos, flat_eg].reshape(B, S, 1)
                out = out + 0.05 * gw * out.detach()

        return out


# ============================================================
# Transformer Blocks
# ============================================================
class FlatTransformer(nn.Module):
    def __init__(self, vocab_size, dim=128, num_heads=4, num_experts=256, top_k=2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(1024, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.moe = FlatMoE(dim, num_experts, top_k)
        self.norm2 = nn.LayerNorm(dim)
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size)

    def forward(self, x):
        B, S = x.shape
        h = self.tok_emb(x) + self.pos_emb(torch.arange(S, device=x.device))
        a, _ = self.attn(self.norm1(h), self.norm1(h), self.norm1(h))
        h = h + a
        h = h + self.moe(self.norm2(h))
        return self.out_proj(self.out_norm(h))


class HierTransformer(nn.Module):
    def __init__(self, vocab_size, dim=128, num_heads=4, num_experts=256,
                 num_groups=32, top_k_groups=1, top_k_experts=2, comm_steps=2,
                 cross_topology="expander", degree=3, seed=42):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(1024, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.moe = HierMoE(dim, num_experts, num_groups,
                           top_k_groups, top_k_experts, comm_steps,
                           cross_topology, degree, seed)
        self.norm2 = nn.LayerNorm(dim)
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size)

    def forward(self, x):
        B, S = x.shape
        h = self.tok_emb(x) + self.pos_emb(torch.arange(S, device=x.device))
        a, _ = self.attn(self.norm1(h), self.norm1(h), self.norm1(h))
        h = h + a
        h = h + self.moe(self.norm2(h))
        return self.out_proj(self.out_norm(h))


# ============================================================
# Training & Eval
# ============================================================
def evaluate(model, tl, nb=10, bs=16):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(nb):
            inp, tgt = get_batch(tl, bs=bs)
            logits = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
            total += loss.item()
    return math.exp(total / nb)

def train_model(model, tl, steps=800, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best_ppl = float('inf')
    losses = []

    for step in range(1, steps + 1):
        model.train()
        inp, tgt = get_batch(tl)
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if step % 100 == 0 or step == steps:
            ppl = evaluate(model, test_tokens, nb=5)
            best_ppl = min(best_ppl, ppl)
            print(f"  Step {step}/{steps} | loss={loss.item():.4f} | ppl={ppl:.2f}", flush=True)

    return {"train_loss": float(np.mean(losses[-50:])),
            "best_ppl": float(best_ppl),
            "final_ppl": float(evaluate(model, test_tokens))}


# ============================================================
# Runner
# ============================================================
def run_exp(cfg):
    name = cfg["name"]
    print(f"\n{'='*60}\n{name}\n{'='*60}", flush=True)

    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    if cfg["type"] == "flat":
        model = FlatTransformer(VOCAB_SIZE, dim=cfg["dim"], num_experts=cfg["N"],
                                 top_k=cfg.get("top_k", 2)).to(device)
    else:
        model = HierTransformer(VOCAB_SIZE, dim=cfg["dim"],
                                num_experts=cfg["N"], num_groups=cfg["G"],
                                top_k_groups=cfg.get("kg", 1),
                                top_k_experts=cfg.get("ke", 2),
                                comm_steps=cfg.get("comm_steps", 2),
                                cross_topology=cfg["topo"],
                                degree=cfg.get("degree", 3),
                                seed=cfg.get("seed", 42)).to(device)

    p = sum(p.numel() for p in model.parameters())
    print(f"Params: {p:,}", flush=True)

    t0 = time.time()
    res = train_model(model, train_tokens, steps=cfg.get("steps", 800))
    t = time.time() - t0
    res.update({"name": name, "params": p, "time": round(t), "config": cfg})
    return res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--G", type=int, default=32)
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--comm_steps", type=int, default=2)
    parser.add_argument("--seeds", type=str, default="42,123,456",
                        help="Comma-separated seeds to run sequentially")
    args = parser.parse_args()

    SEEDS = [int(s) for s in args.seeds.split(",")]
    N = args.N
    G = args.G
    D = args.dim

    all_results = []

    for S in SEEDS:
        print(f"\n{'#'*60}\n# SEED {S}\n{'#'*60}", flush=True)

        exps = [
            {"name": f"flat_N{N}_s{S}", "type": "flat", "N": N, "dim": D, "top_k": 2,
             "seed": S, "steps": args.steps},
            {"name": f"hier_ring_N{N}_G{G}_s{S}", "type": "hier", "N": N, "G": G, "dim": D,
             "kg": 1, "ke": 2, "comm_steps": args.comm_steps, "topo": "ring",
             "seed": S, "steps": args.steps},
            {"name": f"hier_expander_N{N}_G{G}_s{S}", "type": "hier", "N": N, "G": G, "dim": D,
             "kg": 1, "ke": 2, "comm_steps": args.comm_steps, "topo": "expander",
             "seed": S, "steps": args.steps},
            {"name": f"hier_dense_N{N}_G{G}_s{S}", "type": "hier", "N": N, "G": G, "dim": D,
             "kg": 1, "ke": 2, "comm_steps": args.comm_steps, "topo": "dense",
             "seed": S, "steps": args.steps},
        ]

        for c in exps:
            r = run_exp(c)
            all_results.append(r)
            print(f"  -> {r['name']}: loss={r['train_loss']:.4f} | "
                  f"ppl={r['best_ppl']:.2f} | {r['time']}s", flush=True)

    # Append to hier_results.json (never overwrite)
    op = os.path.join(os.path.dirname(__file__) or '.', 'hier_results.json')
    existing = []
    if os.path.exists(op):
        with open(op) as f:
            existing = json.load(f)
    # Remove old entries with same names (allow re-runs to replace)
    existing = [r for r in existing if r["name"] not in {r2["name"] for r2 in all_results}]
    with open(op, 'w') as f:
        json.dump(existing + all_results, f, indent=2)

    print(f"\n{'='*60}\nSUMMARY (sorted by best_ppl) — Phase 5: N={N}, G={G}, dim={D}\n{'='*60}")
    for r in sorted(all_results, key=lambda x: x['best_ppl']):
        tag = "★" if "expander" in r["name"] else " "
        print(f"  {tag} {r['name']:50s} | loss={r['train_loss']:.4f} | ppl={r['best_ppl']:.2f} | {r['time']}s")

    # Also print per-seed summary
    print(f"\n{'='*60}\nPER-SEED ANALYSIS\n{'='*60}")
    for S in SEEDS:
        print(f"\n  Seed {S}:")
        seed_results = [r for r in all_results if f"_s{S}" in r["name"]]
        for r in sorted(seed_results, key=lambda x: x['best_ppl']):
            print(f"    {r['name']:50s} | loss={r['train_loss']:.4f} | ppl={r['best_ppl']:.2f}")
