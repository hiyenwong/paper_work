"""
Phase 4: Hierarchical MoE with Topology Routing (H-MoE-Topo)
=============================================================
Architecture: N experts in G groups, two-level gating + topology propagation.

Key innovations:
1. Two-level routing: Group gate -> Expert gate (hierarchical sparsity)
2. Expert state communication through two-level topology:
   - Within-group: dense (complete graph, fast local mix)
   - Cross-group: expander (random regular, efficient global mix)
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
    else:  # "none" fallback
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
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, S, D = x.shape
        xf = x.reshape(-1, D)
        gw = F.softmax(self.gate(xf), dim=-1)
        tw, ti = torch.topk(gw, self.top_k, dim=-1)
        tw = tw / (tw.sum(-1, keepdim=True) + 1e-8)

        out = torch.zeros_like(xf)
        for e in range(self.num_experts):
            mask = (ti == e)
            if not mask.any():
                continue
            tids = torch.unique(torch.nonzero(mask)[:, 0])
            w = tw.clone()
            w[~mask] = 0.0
            ws = w.sum(-1)[tids].unsqueeze(-1)
            inp = xf[tids]
            h = F.relu(inp @ self.w1[e] + self.b1[e])
            h = h @ self.w2[e] + self.b2[e]
            out.index_add_(0, tids, h * ws)
        return self.out(out.reshape(B, S, D))


class FlatTransformer(nn.Module):
    def __init__(self, vocab_size, dim=64, num_heads=4, num_experts=64, top_k=2):
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


# ============================================================
# Hierarchical MoE (方案五)
# ============================================================
class HierMoE(nn.Module):
    """N experts, G groups, two-level routing + topology state propagation."""
    def __init__(self, dim, num_experts, num_groups,
                 top_k_groups=1, top_k_experts=2, comm_steps=2,
                 cross_topology="expander", degree=3, seed=42):
        super().__init__()
        assert num_experts % num_groups == 0
        self.N = num_experts
        self.G = num_groups
        self.E = num_experts // num_groups  # experts per group
        self.kg = top_k_groups
        self.ke = top_k_experts
        self.comm_steps = comm_steps

        # Expert FFN parameters
        self.w1 = nn.Parameter(torch.randn(num_experts, dim, dim * 2) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(num_experts, dim * 2))
        self.w2 = nn.Parameter(torch.randn(num_experts, dim * 2, dim) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(num_experts, dim))

        # Persistent expert state vectors
        self.state = nn.Parameter(torch.randn(num_experts, dim) * 0.02)

        # State propagation: within-group (dense)
        I = torch.eye(self.E)
        O = torch.ones(self.E, self.E)
        A_in = O + I
        self.inner_A = (torch.diag(1.0 / A_in.sum(dim=1)) @ A_in).unsqueeze(0)  # (1, E, E)
        self.register_buffer('inner_A_buf', self.inner_A)

        # State propagation: cross-group (expander/ring/dense)
        G = build_cross_graph(cross_topology, self.G, degree, seed)
        A = torch.tensor(nx.adjacency_matrix(G).todense(), dtype=torch.float32)
        A = A + torch.eye(self.G)
        self.cross_A = torch.diag(1.0 / A.sum(dim=1)) @ A  # (G, G)
        self.register_buffer('cross_A_buf', self.cross_A)

        # State transform weights
        self.W_self = nn.Linear(dim, dim, bias=False)
        self.W_nei = nn.Linear(dim, dim, bias=False)

        # State-conditioned gating: project normalized state to key space
        self.state_to_group = nn.Linear(dim, dim, bias=False)
        self.state_to_expert = nn.Linear(dim, dim, bias=False)
        # Normalize states before conditioning so topology signal is O(1) not O(0.02)
        self.state_norm = nn.LayerNorm(dim)
        # Learnable scale for FFN state conditioning (starts small, grows if useful)
        self.state_scale = nn.Parameter(torch.ones(1) * 0.1)
        self._dim = dim

        # Gating
        self.gate_g = nn.Linear(dim, self.G)
        self.gate_e = nn.Linear(dim, self.E)
        self.out_proj = nn.Linear(dim, dim)

        # Stats
        self.cross_diam = nx.diameter(G) if nx.is_connected(G) else -1
        self.cross_avg = nx.average_shortest_path_length(G) if nx.is_connected(G) else -1

    def propagate_state(self):
        """Two-level state propagation with topology-aware message passing."""
        s = self.state  # (N, D)
        D = s.shape[-1]
        for _ in range(self.comm_steps):
            # Reshape: (N, D) -> (G, E, D)
            s2 = s.view(self.G, self.E, -1)
            # Within-group dense mix
            s_in = torch.einsum('eE,gEd->gEd', self.inner_A_buf[0], s2)  # (G, E, D)
            # Cross-group centroid exchange — normalize first so W_self/W_nei get gradients
            cent = F.layer_norm(s_in.mean(dim=1), [D])   # (G, D) normalized
            cent_m = self.cross_A_buf @ cent              # (G, D) topology-weighted neighbors
            # GELU: no dead neurons unlike relu; W_nei carries the topology gradient path
            cent_new = F.gelu(self.W_self(cent) + self.W_nei(cent_m))
            # Broadcast topology update to all experts in each group
            diff = cent_new.unsqueeze(1) * 0.3  # (G, 1, D)
            s = (s_in + diff).reshape(self.N, -1)
        return s

    def forward(self, x):
        B, S, D = x.shape
        N = B * S
        xf = x.reshape(-1, D)

        # Propagate + normalize: ensures topology signal is O(1), not swamped by gate_g(xf)
        mixed_states = self.state_norm(self.propagate_state())  # (N_experts, D)

        # Group-level state conditioning: token × normalized_group_state dot-product
        s_g = mixed_states.view(self.G, self.E, -1).mean(dim=1)     # (G, D)
        keys_g = self.state_to_group(s_g)                            # (G, D)
        g_condition = xf @ keys_g.T / math.sqrt(self._dim)          # (N_tok, G)

        # Group gate: input features + topology-conditioned per-token logit
        group_logits = self.gate_g(xf) + g_condition                 # (N_tok, G)
        gg = F.softmax(group_logits, dim=-1)
        gw, gi = torch.topk(gg, self.kg, dim=-1)
        gw = gw / (gw.sum(-1, keepdim=True) + 1e-8)

        output = torch.zeros(N, D, device=x.device)

        for g in range(self.G):
            # Tokens assigned to this group
            mask = (gi == g)
            if not mask.any():
                continue
            tids = torch.unique(torch.nonzero(mask)[:, 0])

            # Expert-level state conditioning: token × expert_state dot-product
            s_e = mixed_states[g * self.E : (g + 1) * self.E]        # (E, D)
            keys_e = self.state_to_expert(s_e)                        # (E, D)
            e_condition = xf[tids] @ keys_e.T / math.sqrt(self._dim) # (n_t, E)

            # Expert gate: input features + topology-conditioned per-token logit
            e_logits = self.gate_e(xf[tids]) + e_condition            # (n_t, E)
            eg = F.softmax(e_logits, dim=-1)
            ew, ei = torch.topk(eg, self.ke, dim=-1)
            ew = ew / (ew.sum(-1, keepdim=True) + 1e-8)

            # Global expert IDs
            ge = g * self.E + ei  # (n, ke)

            n_t = tids.shape[0]
            tok_out = torch.zeros(n_t, D, device=x.device)

            # Per-token loop (n_t is typically small)
            for i in range(n_t):
                inp = xf[tids[i]]  # (D,)
                for k in range(self.ke):
                    e = ge[i, k].item()
                    w = ew[i, k]
                    # Condition FFN input with topology-shaped expert state
                    # Gradient: loss → h → inp_mod → mixed_states[e] → cross_A_buf
                    inp_mod = inp + mixed_states[e] * self.state_scale
                    h = F.relu(inp_mod @ self.w1[e] + self.b1[e])  # (2D,)
                    h = h @ self.w2[e] + self.b2[e]  # (D,)
                    tok_out[i] = tok_out[i] + h * w

            output.index_add_(0, tids, tok_out)

        return self.out_proj(output.reshape(B, S, D))


class HierTransformer(nn.Module):
    def __init__(self, vocab_size, dim=64, num_heads=4, num_experts=64,
                 num_groups=8, top_k_groups=1, top_k_experts=2, comm_steps=2,
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

def train_model(model, tl, steps=300, lr=3e-4):
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

        if step % 50 == 0 or step == steps:
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
    res = train_model(model, train_tokens, steps=cfg.get("steps", 300))
    t = time.time() - t0
    res.update({"name": name, "params": p, "time": round(t), "config": cfg})
    return res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--G", type=int, default=16, help="Number of groups")
    parser.add_argument("--N", type=int, default=64, help="Total experts")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--comm_steps", type=int, default=2)
    args = parser.parse_args()

    S = args.seed
    N = args.N
    G = args.G
    D = 64

    print(f"Config: N={N}, G={G}, E={N//G}, steps={args.steps}, comm_steps={args.comm_steps}, seed={S}", flush=True)

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

    results = []
    for c in exps:
        r = run_exp(c)
        results.append(r)
        print(f"  -> {r['name']}: loss={r['train_loss']:.4f} | "
              f"ppl={r['best_ppl']:.2f} | {r['time']}s", flush=True)

    # Append to hier_results.json (never overwrite)
    op = os.path.join(os.path.dirname(__file__) or '.', 'hier_results.json')
    existing = []
    if os.path.exists(op):
        with open(op) as f:
            existing = json.load(f)
    # Replace entries with same name, append new ones
    existing_names = {r["name"] for r in existing}
    existing = [r for r in existing if r["name"] not in {r2["name"] for r2 in results}]
    with open(op, 'w') as f:
        json.dump(existing + results, f, indent=2)

    print(f"\n{'='*60}\nSUMMARY (sorted by best_ppl)\n{'='*60}")
    for r in sorted(results, key=lambda x: x['best_ppl']):
        tag = "★" if "expander" in r["name"] else " "
        print(f"  {tag} {r['name']:40s} | loss={r['train_loss']:.4f} | ppl={r['best_ppl']:.2f} | {r['time']}s")
