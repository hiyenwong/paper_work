"""
graph_signal_propagation.py
===========================
Experiment: Learn to propagate smooth graph signals.

Key design insight from failures:
- Previous: random signals (no structure → nothing to propagate)
- This: signals are SMOOTH on the graph (nearby experts = similar targets)
- The model must learn to use the topology for propagation

Experiment setup:
1. N=64 experts arranged on a graph
2. Targets: each expert has a target that is correlated with graph distance
3. Training: each expert starts with a random guess, must refine via message passing
4. The topology determines how efficiently information spreads

Expected hierarchy (higher is better):
  Dense > Random Regular > Ring > None
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import math
import time
import json
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")


def build_graph(name: str, N: int, d: int):
    np.random.seed(42)
    if name == "ring":
        assert d % 2 == 0
        G = nx.Graph()
        for i in range(N):
            for off in range(1, d // 2 + 1):
                G.add_edge(i, (i + off) % N)
    elif name == "random_regular":
        G = nx.random_regular_graph(d, N, seed=42)
    elif name == "dense":
        G = nx.complete_graph(N)
    else:
        raise ValueError(name)
    return G


def generate_smooth_targets(N, D, graph_name, degree=4, tau=2.0):
    """
    Generate target signals that are smooth on the graph.
    
    Approach: start with random Gaussian noise per node,
    then diffuse through the graph laplacian to create smooth signals.
    Nearby nodes will have correlated targets.
    """
    if graph_name == "none":
        return torch.randn(N, D) * 0.1
    
    G = build_graph(graph_name, N, degree)
    rng = np.random.RandomState(42)
    
    # Initial random signals
    base = rng.randn(N, D)
    
    # Diffuse through graph to create smoothness
    A = nx.adjacency_matrix(G).todense()
    D_mat = np.diag(np.array(A.sum(axis=1)).flatten())
    L = D_mat - A  # Laplacian
    # Apply heat kernel: exp(-tau * L)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    heat_kernel = eigenvectors @ np.diag(np.exp(-tau * eigenvalues)) @ eigenvectors.T
    targets = heat_kernel @ base
    
    # Normalize
    targets = targets / (np.std(targets) + 1e-8)
    
    return torch.tensor(targets, dtype=torch.float32)


class GraphSignalPropagator(nn.Module):
    """
    Learn to propagate signals through a graph topology.
    
    Each expert has a learnable embedding. At each step, experts
    exchange information through the graph edges.
    The readout predicts the smooth target signal.
    """
    def __init__(self, N, D, topology, degree=4, num_steps=8):
        super().__init__()
        self.N = N
        self.D = D
        self.num_steps = num_steps
        
        # Learnable initial embeddings (per expert)
        self.embed = nn.Parameter(torch.randn(N, D) * 0.1)
        
        # Shared message passing weights (more efficient than per-expert)
        self.W_self = nn.Linear(D, D)
        self.W_neighbor = nn.Linear(D, D)
        
        # Normalization
        self.norm = nn.LayerNorm(D)
        
        # Readout
        self.readout = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, D),
        )
        
        if topology != "none":
            G = build_graph(topology, N, degree)
            A = torch.tensor(nx.adjacency_matrix(G).todense(), dtype=torch.float32)
            # Random walk normalization
            D_inv = torch.diag(1.0 / A.sum(dim=1).clamp(min=1))
            self.A_norm = A @ D_inv  # (D^-1 * A)
            self.diameter = nx.diameter(G) if nx.is_connected(G) else -1
        else:
            self.A_norm = torch.eye(N)
            self.diameter = -1
        
        self.graph_name = topology
    
    def forward(self, x=None):
        """
        x: optional initial signals (None = use learned embeddings)
        Returns: (N, D) predicted signals
        """
        h = self.embed if x is None else x
        
        # Batched message passing
        A = self.A_norm.to(h.device)
        for step in range(self.num_steps):
            # Neighbor aggregation: A @ h
            neighbor_sum = A @ h  # (N, D)
            
            # Update: h_new = norm( W_self @ h + W_neighbor @ neighbor_sum )
            self_update = self.W_self(h)
            neighbor_update = self.W_neighbor(neighbor_sum)
            h = self.norm(torch.tanh(self_update + neighbor_update))
        
        return self.readout(h)


@torch.no_grad()
def compute_graph_stats(name, N, d=4):
    G = build_graph(name, N, d)
    return {
        "diameter": nx.diameter(G) if nx.is_connected(G) else -1,
        "avg_path": nx.average_shortest_path_length(G),
    }


def run_experiment(topology, N=64, D=32, shared_targets=None,
                   degree=4, steps=500, lr=3e-4, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Use shared targets (generated from random_regular for all experiments)
    # This ensures fair comparison: same targets, different message passing graphs
    if shared_targets is None:
        targets = generate_smooth_targets(N, D, "random_regular", degree)
    else:
        targets = shared_targets
    targets = targets.to(device)
    
    model = GraphSignalPropagator(N, D, topology, degree).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    start = time.time()
    losses = []
    
    for step in range(steps):
        # Predict using learned embeddings + message passing
        pred = model()
        loss = F.mse_loss(pred, targets)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        losses.append(loss.item())
        
        if step % 100 == 0:
            elapsed = time.time() - start
            print(f"  Step {step:4d}: loss={loss.item():.6f} ({elapsed:.0f}s)", flush=True)
    
    final_loss = np.mean(losses[-50:])
    best_loss = np.min(losses[-100:])
    elapsed = time.time() - start
    
    print(f"  [{topology}] FINAL loss={final_loss:.6f}, best={best_loss:.6f}, "
          f"diam={model.diameter} ({elapsed:.0f}s)", flush=True)
    
    return {
        "topology": topology,
        "final_loss": round(final_loss, 6),
        "best_loss": round(best_loss, 6),
        "diameter": model.diameter if model.diameter >= 0 else "N/A",
        "time": round(elapsed),
    }


def main():
    print("=" * 70, flush=True)
    print("GRAPH SIGNAL PROPAGATION (Smooth Targets)")
    print("=" * 70, flush=True)
    
    N, D = 64, 32
    
    print(f"\nN={N}, D={D}")
    for name in ["ring", "random_regular", "dense"]:
        s = compute_graph_stats(name, N)
        print(f"  {name:<16} diam={s['diameter']:>3}, avg_path={s['avg_path']:.2f}")
    
    # Quick sanity: show target smoothness
    for name in ["ring", "random_regular", "dense"]:
        targets = generate_smooth_targets(N, D, name)
        # Correlation between connected nodes
        G = build_graph(name, N, 4)
        corrs = []
        for (i, j) in G.edges():
            c = F.cosine_similarity(targets[i:i+1], targets[j:j+1]).item()
            corrs.append(c)
        print(f"  {name:<16} neighbor_corr={np.mean(corrs):.3f} (+-{np.std(corrs):.3f})")
    
    # Generate shared targets (from random_regular for fairness)
    shared_targets = generate_smooth_targets(N, D, "random_regular")
    
    results = []
    for topo in ["none", "ring", "random_regular", "dense"]:
        print(f"\n--- {topo} ---", flush=True)
        r = run_experiment(topo, N=N, D=D, shared_targets=shared_targets, steps=500)
        results.append(r)
    
    print(f"\n{'='*70}", flush=True)
    print("FINAL RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Topology':<18} {'Final Loss':<14} {'Best Loss':<14} {'Diameter':<10}", flush=True)
    print("-" * 56, flush=True)
    for r in results:
        print(f"{r['topology']:<18} {r['final_loss']:<14} {r['best_loss']:<14} "
              f"{str(r['diameter']):<10}", flush=True)
    print("-" * 56, flush=True)
    
    base = [r for r in results if r['topology'] == 'none'][0]['best_loss']
    ring_r = [r for r in results if r['topology'] == 'ring'][0]['best_loss']
    reg_r = [r for r in results if r['topology'] == 'random_regular'][0]['best_loss']
    dense_r = [r for r in results if r['topology'] == 'dense'][0]['best_loss']
    
    print(f"\n  None (no comm): {base}", flush=True)
    print(f"  vs Ring:           {ring_r}  ({ring_r/base-1:+.1%})", flush=True)
    print(f"  vs Random Regular: {reg_r}  ({reg_r/base-1:+.1%})", flush=True)
    print(f"  vs Dense:          {dense_r}  ({dense_r/base-1:+.1%})", flush=True)
    
    checks = [
        ("Ring < None (comm helps)", ring_r < base),
        ("Random Reg < Ring (better topo better)", reg_r < ring_r),
        ("Dense <= Random Reg (richest best)", dense_r <= reg_r),
        ("Dense < None (any comm > no comm)", dense_r < base),
    ]
    for desc, ok in checks:
        print(f"  {'✅' if ok else '❌'} {desc}", flush=True)
    
    out = os.path.join(os.path.dirname(__file__) or '.', 'smooth_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}", flush=True)
    print(f"{'='*70}\n", flush=True)


if __name__ == "__main__":
    main()
