"""
Phase 2B v2: Proper Tiny MoE Training (MLX)
=============================================
Using real text (Alice in Wonderland, 78K chars)
Sparse soft MoE with proper train/test split
500 steps per topology
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import networkx as nx
import math
import time
import sys


# ============================================================
# 1. Char-level Tokenizer
# ============================================================
# Get all characters from the training text
with open('/tmp/alice_clean.txt') as f:
    ALICE_TEXT = f.read()

CHARS = sorted(list(set(ALICE_TEXT)))
VOCAB_SIZE = len(CHARS) + 4
char_to_id = {c: i + 4 for i, c in enumerate(CHARS)}
char_to_id['<bos>'] = 0
char_to_id['<eos>'] = 1
char_to_id['<pad>'] = 2
char_to_id['<unk>'] = 3
id_to_char = {v: k for k, v in char_to_id.items()}


def encode(text: str) -> list:
    return [char_to_id.get(c, 3) for c in text]


# Train/test split (80/20)
tokens = encode(ALICE_TEXT)
split_idx = int(len(tokens) * 0.8)
train_tokens = tokens[:split_idx]
test_tokens = tokens[split_idx:]
print(f"Train: {len(train_tokens)} chars, Test: {len(test_tokens)} chars")


def generate_batch(tokens_list, seq_len: int = 64, batch_size: int = 32, seed: int = None):
    """Generate batches from token list."""
    if seed is not None:
        np.random.seed(seed)
    L = len(tokens_list)
    inputs, targets = [], []
    for _ in range(batch_size):
        start = np.random.randint(0, max(1, L - seq_len - 1))
        inp = tokens_list[start:start + seq_len]
        tgt = tokens_list[start + 1:start + seq_len + 1]
        if len(inp) < seq_len:
            inp = inp + [2] * (seq_len - len(inp))
            tgt = tgt + [2] * (seq_len - len(tgt))
        inputs.append(inp)
        targets.append(tgt)
    return mx.array(inputs), mx.array(targets)


# ============================================================
# 2. Graph Topology
# ============================================================
def build_graph(name: str, N: int, d: int, seed: int = 42):
    np.random.seed(seed)
    if name == "ring":
        assert d % 2 == 0
        G = nx.Graph()
        G.add_nodes_from(range(N))
        for i in range(N):
            for offset in range(1, d // 2 + 1):
                G.add_edge(i, (i + offset) % N)
    elif name == "random_regular":
        G = nx.random_regular_graph(d, N, seed=seed)
    elif name == "dense":
        G = nx.complete_graph(N)
    else:
        raise ValueError(f"Unknown: {name}")
    adj = {i: list(G.neighbors(i)) for i in range(N)}
    return adj, G


def make_msg_matrix(adj_list: dict, d: int, N: int):
    rows = []
    for i in range(N):
        row = mx.zeros(N)
        row = row.at[i].add(1.0 / (d + 1))
        for j in adj_list.get(i, []):
            row = row.at[j].add(1.0 / (d + 1))
        rows.append(row)
    return mx.stack(rows)


def compute_topology_metrics(name: str, N: int, d: int):
    if name == "none":
        return {"diameter": "N/A", "connected": True}
    _, G = build_graph(name, N, d)
    if not nx.is_connected(G):
        return {"diameter": float('inf'), "connected": False}
    return {
        "diameter": nx.diameter(G),
        "avg_path": nx.average_shortest_path_length(G),
        "connected": True,
    }


# ============================================================
# 3. MoE Model
# ============================================================
class Expert(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.ReLU(),
            nn.Linear(dim * hidden_mult, dim),
        )
    
    def __call__(self, x: mx.array) -> mx.array:
        return self.net(x)


class SoftMoELayer(nn.Module):
    """Soft MoE: all experts process all tokens, weighted by top-k gate.
    
    This is MLX-compatible (no scatter/gather needed).
    After expert computation, states are mixed through topology.
    """
    def __init__(self, dim: int, num_experts: int, top_k: int,
                 topology_name: str, degree: int):
        super().__init__()
        self.num_experts = num_experts
        self.num_active = top_k
        self.topology = topology_name
        
        self.experts = [Expert(dim) for _ in range(num_experts)]
        self.gate = nn.Linear(dim, num_experts)
        
        if topology_name != "none":
            adj_list, G = build_graph(topology_name, num_experts, degree)
            self.msg_matrix = make_msg_matrix(adj_list, degree, num_experts)
        else:
            self.msg_matrix = None
        
        self.out_proj = nn.Linear(dim, dim)
    
    def __call__(self, x: mx.array) -> mx.array:
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)
        N = B * S
        
        # Gate
        gate_logits = self.gate(x_flat)
        gate_weights = mx.softmax(gate_logits, axis=-1)
        
        # Top-k routing (detached from gradients)
        gw = mx.stop_gradient(gate_weights)
        sorted_inds = mx.argsort(gw, axis=-1)
        gate_mask = mx.zeros_like(gw)
        gate_mask = gate_mask.at[:, sorted_inds[:, -self.num_active:]].add(1.0)
        
        # Masked weights (re-attach to continuous gate_weights for gradient flow)
        gate_masked = gate_weights * gate_mask
        gate_masked = gate_masked / (mx.sum(gate_masked, axis=-1, keepdims=True) + 1e-8)
        
        # All experts process all tokens
        expert_stack = mx.stack([expert(x_flat) for expert in self.experts])  # (E, N, D)
        
        # Weight by gate (detach routing weights from gradients)
        weights = mx.stop_gradient(gate_masked.T[:, :, None])
        weighted = expert_stack * weights
        
        # Topology communication
        if self.topology != "none":
            # (E, N, D) -> (N, D, E) @ (E, E) -> (E, N, D)
            comm = weighted.transpose(1, 2, 0) @ self.msg_matrix.T
            aggregated = comm.transpose(2, 0, 1)
        else:
            aggregated = weighted
        
        out = mx.sum(aggregated, axis=0).reshape(B, S, D)
        return self.out_proj(out)


class MoETransformer(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 64, num_heads: int = 4,
                 num_layers: int = 2, num_experts: int = 64, top_k: int = 4,
                 topology: str = "none", degree: int = 4):
        super().__init__()
        self.dim = dim
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(1024, dim)
        
        self.layers = []
        for i in range(num_layers):
            attn = nn.MultiHeadAttention(dim, num_heads)
            moe = SoftMoELayer(dim, num_experts, top_k, topology, degree)
            norm1 = nn.RMSNorm(dim)
            norm2 = nn.RMSNorm(dim)
            setattr(self, f'attn_{i}', attn)
            setattr(self, f'moe_{i}', moe)
            setattr(self, f'norm1_{i}', norm1)
            setattr(self, f'norm2_{i}', norm2)
            self.layers.append({
                'attn': attn, 'norm1': norm1,
                'moe': moe, 'norm2': norm2,
            })
        
        self.out_norm = nn.RMSNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size)
    
    def __call__(self, x: mx.array) -> mx.array:
        B, S = x.shape
        h = self.token_embed(x) + self.pos_embed(mx.arange(S)[None, :])
        
        for layer in self.layers:
            # Self-attention
            attn_out = layer['attn'](layer['norm1'](h), layer['norm1'](h), layer['norm1'](h))
            h = h + attn_out
            # MoE
            moe_out = layer['moe'](layer['norm2'](h))
            h = h + moe_out
        
        h = self.out_norm(h)
        return self.out_proj(h)


# ============================================================
# 4. Training
# ============================================================
def loss_fn(model, inputs, targets):
    logits = model(inputs)
    B, S, V = logits.shape
    return nn.losses.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction='mean')


def train_one_batch(model, optimizer, inputs, targets):
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, inputs, targets)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return loss


def evaluate_ppl(model, tokens_list, seq_len: int = 64, num_batches: int = 10):
    """Evaluate perplexity on held-out data, averaging over multiple batches."""
    total_nll = 0.0
    count = 0
    
    for b in range(num_batches):
        inputs, targets = generate_batch(tokens_list, seq_len=seq_len, 
                                          batch_size=16, seed=b)
        logits = model(inputs)
        B, S, V = logits.shape
        loss = nn.losses.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), 
                                        reduction='mean')
        total_nll += loss.item()
        count += 1
    
    avg_nll = total_nll / count
    return math.exp(avg_nll)


def run_experiment(topology: str, num_experts: int = 64, degree: int = 4,
                   dim: int = 64, num_heads: int = 4, num_layers: int = 2,
                   top_k: int = 4, num_steps: int = 500, lr: float = 3e-4,
                   seed: int = 42) -> dict:
    
    mx.random.seed(seed)
    np.random.seed(seed)
    
    # Build model
    model = MoETransformer(
        vocab_size=VOCAB_SIZE, dim=dim, num_heads=num_heads,
        num_layers=num_layers, num_experts=num_experts, top_k=top_k,
        topology=topology, degree=degree,
    )
    mx.eval(model.parameters())
    
    optimizer = optim.Adam(learning_rate=lr)
    
    start = time.time()
    print(f"\n  [{topology}] N={num_experts}, d={degree}, {dim}dim, {num_layers}L")
    
    for step in range(num_steps):
        inputs, targets = generate_batch(train_tokens, seq_len=64, batch_size=32, 
                                          seed=step)
        loss = train_one_batch(model, optimizer, inputs, targets)
        
        if step % 100 == 0:
            train_ppl = evaluate_ppl(model, train_tokens[:5000], num_batches=5)
            test_ppl = evaluate_ppl(model, test_tokens, num_batches=5)
            elapsed = time.time() - start
            print(f"    Step {step:3d}: loss={loss.item():.3f} train_ppl={train_ppl:.1f} "
                  f"test_ppl={test_ppl:.1f} ({elapsed:.0f}s)")
    
    # Final evaluation
    final_train_ppl = evaluate_ppl(model, train_tokens[:5000], num_batches=10)
    final_test_ppl = evaluate_ppl(model, test_tokens, num_batches=10)
    elapsed = time.time() - start
    
    print(f"  [{topology}] Done: train_ppl={final_train_ppl:.1f} "
          f"test_ppl={final_test_ppl:.1f} ({elapsed:.0f}s)")
    
    metrics = compute_topology_metrics(topology, num_experts, degree)
    
    return {
        "topology": topology,
        "train_ppl": round(final_train_ppl, 2),
        "test_ppl": round(final_test_ppl, 2),
        "time_sec": round(elapsed),
        "diameter": metrics.get("diameter", "?"),
        "connected": metrics.get("connected", False),
    }


def main():
    print("=" * 80)
    print("PHASE 2B v2: TINY MoE TRAINING (Alice in Wonderland, 78K chars)")
    print("=" * 80)
    
    CONFIG = {
        "num_experts": 64,
        "degree": 4,
        "dim": 32,
        "num_heads": 4,
        "num_layers": 1,
        "top_k": 4,
        "num_steps": 300,
        "lr": 3e-4,
    }
    
    print(f"\nArchitecture: {CONFIG['num_layers']}L Transformer + MoE")
    print(f"  Dim={CONFIG['dim']}, Heads={CONFIG['num_heads']}")
    print(f"  Experts={CONFIG['num_experts']}, Top-K={CONFIG['top_k']}, d={CONFIG['degree']}")
    print(f"  Vocab={VOCAB_SIZE}, Steps={CONFIG['num_steps']}")
    print(f"  Dataset: {len(train_tokens)} train + {len(test_tokens)} test chars")
    
    # Print topology metrics first
    print(f"\n{'='*60}")
    print("Topology properties (N=64, d=4):")
    for topo in ["ring", "random_regular", "dense"]:
        m = compute_topology_metrics(topo, 64, 4)
        if m["connected"]:
            print(f"  {topo:<16} diameter={m['diameter']:>3} avg_path={m['avg_path']:.2f}")
        else:
            print(f"  {topo:<16} DISCONNECTED")
    
    # Run experiments
    topologies = ["none", "ring", "random_regular", "dense"]
    results = []
    for topo in topologies:
        r = run_experiment(topology=topo, **CONFIG)
        results.append(r)
    
    # Summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"{'Topology':<18} {'Train PPL':<12} {'Test PPL':<12} {'Diameter':<10} {'Time':<8}")
    print("-" * 60)
    for r in results:
        print(f"{r['topology']:<18} {r['train_ppl']:<12} {r['test_ppl']:<12} "
              f"{r['diameter']:<10} {r['time_sec']:<8}")
    print("-" * 60)
    
    # Find best
    best = min(results, key=lambda r: r['test_ppl'])
    worst = max(results, key=lambda r: r['test_ppl'])
    baseline = [r for r in results if r['topology'] == 'none'][0]['test_ppl']
    
    print(f"\n  Best:  {best['topology']}  test_ppl={best['test_ppl']}")
    print(f"  Worst: {worst['topology']}  test_ppl={worst['test_ppl']}")
    print(f"  Baseline (none): test_ppl={baseline}")
    print(f"\n  If expander (random_regular) outperforms ring AND dense:")
    print(f"    → Confirms Ramanujan topology benefit")
    print(f"  If dense is best:")
    print(f"    → Confirms richer comm helps, expander may need more experts")
    print(f"  If ring is worst:")
    print(f"    → Confirms bad topology hurts MoE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
