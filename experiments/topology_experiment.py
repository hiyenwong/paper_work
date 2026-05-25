"""
topology_experiment.py
======================
Experiment: Topology-Constrained Expert Communication

N experts with learnable embeddings communicate through a topology graph.
The model must learn to propagate signals through the graph.

This is the simplest setting where topology should matter.
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


# ============================================================
# Graph builders
# ============================================================
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


class TopologyConstrainedExpert(nn.Module):
    """
    N experts with learnable representations.
    
    Forward: each expert sees its own input AND aggregated neighbor states.
    The topology determines which experts exchange information.
    """
    def __init__(self, N: int, D: int, topology: str, degree: int = 4, 
                 num_steps: int = 5):
        super().__init__()
        self.N = N
        self.D = D
        self.topology = topology
        self.num_steps = num_steps
        
        # Learnable expert embeddings
        self.expert_embed = nn.Parameter(torch.randn(N, D) * 0.1)
        
        # Per-expert transformation
        self.W = nn.Parameter(torch.randn(N, D, D) * 0.1)
        self.b = nn.Parameter(torch.zeros(N, D))
        
        # Node-specific readout
        self.readout = nn.Linear(D, D)
        
        if topology != "none":
            G = build_graph(topology, N, degree)
            # Normalized adjacency
            A = torch.tensor(nx.adjacency_matrix(G).todense(), dtype=torch.float32)
            D_inv_sqrt = torch.diag(1.0 / A.sum(dim=1).clamp(min=1).sqrt())
            self.A_hat = D_inv_sqrt @ A @ D_inv_sqrt  # symmetric norm
        else:
            self.A_hat = torch.eye(N)
        
        # Graph stats
        if topology != "none" and N <= 1000:
            self.graph_diameter = nx.diameter(G) if nx.is_connected(G) else -1
        else:
            self.graph_diameter = -1
    
    def forward(self, x):
        """
        x: (batch, N, D) - initial signals for each expert
        Returns: (batch, N, D) - propagated signals
        """
        B = x.shape[0]
        
        if self.topology == "none":
            # No communication: each expert processes independently
            outs = []
            for i in range(self.N):
                outs.append(x[:, i] @ self.W[i] + self.b[i])
            h = torch.stack(outs, dim=1)
            return self.readout(h)
        
        # Multi-step message passing through topology
        h = x
        for step in range(self.num_steps):
            # Update each expert: combine its own state with neighbors
            neighbor_agg = torch.einsum('ij,bjd->bid', self.A_hat.to(x.device), h)
            
            # Per-expert transformation (no in-place ops!)
            h_new_list = []
            for i in range(self.N):
                own = h[:, i] @ self.W[i] + self.b[i]
                nb = neighbor_agg[:, i]
                h_new_list.append(torch.tanh(own + nb))
            h = torch.stack(h_new_list, dim=1)  # (B, N, D)
        
        return self.readout(h)


def run_experiment(topology: str, N: int = 64, D: int = 32, degree: int = 4,
                   batch_size: int = 64, steps: int = 500, lr: float = 1e-3):
    """Run one topology experiment."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = TopologyConstrainedExpert(N, D, topology, degree).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    start = time.time()
    losses = []
    
    for step in range(steps):
        # Generate random target signals
        # Each batch: B random signals at B random expert nodes
        signals = torch.randn(batch_size, N, D, device=device)
        
        # Input: only the signal at the source expert, zeros elsewhere
        source = torch.randint(0, N, (batch_size,), device=device)
        x = torch.zeros(batch_size, N, D, device=device)
        for b in range(batch_size):
            x[b, source[b]] = signals[b, source[b]]
        
        # Forward: propagate through topology
        pred = model(x)
        
        # Loss: MSE between prediction and true signals
        loss = F.mse_loss(pred, signals)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        losses.append(loss.item())
        
        if step % 100 == 0:
            elapsed = time.time() - start
            print(f"  Step {step:4d}: loss={loss.item():.6f} ({elapsed:.0f}s)")
    
    final_loss = np.mean(losses[-50:])
    elapsed = time.time() - start
    diameter = model.graph_diameter
    
    print(f"  [{topology}] FINAL loss={final_loss:.6f}, diam={diameter} ({elapsed:.0f}s)")
    
    return {
        "topology": topology,
        "final_loss": round(final_loss, 6),
        "diameter": diameter if diameter > 0 else "N/A",
        "time": round(elapsed),
        "loss_trace": losses[::10],  # Sample every 10 steps
    }


def main():
    print("=" * 70)
    print("TOPOLOGY-CONSTRAINED EXPERT COMMUNICATION")
    print("=" * 70)
    
    N, D = 64, 32
    
    # Print topology properties
    print(f"\nTopology properties (N={N}, d=4):")
    for name in ["ring", "random_regular", "dense"]:
        G = build_graph(name, N, 4)
        d = nx.diameter(G)
        ap = nx.average_shortest_path_length(G)
        print(f"  {name:<16} diam={d:>3}, avg_path={ap:.2f}")
    
    results = []
    for topo in ["none", "ring", "random_regular", "dense"]:
        print(f"\n--- {topo} ---")
        r = run_experiment(topo, N=N, D=D, steps=500)
        results.append(r)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"{'Topology':<18} {'Final Loss':<15} {'Diameter':<10}")
    print("-" * 43)
    for r in results:
        print(f"{r['topology']:<18} {r['final_loss']:<15} {str(r['diameter']):<10}")
    print("-" * 43)
    
    base = [r for r in results if r['topology'] == 'none'][0]['final_loss']
    ring_r = [r for r in results if r['topology'] == 'ring'][0]['final_loss']
    reg_r = [r for r in results if r['topology'] == 'random_regular'][0]['final_loss']
    dense_r = [r for r in results if r['topology'] == 'dense'][0]['final_loss']
    
    print(f"\n  None (no comm): {base}")
    print(f"  Ring:           {ring_r}  ({ring_r/base-1:+.1%})")
    print(f"  Random Regular: {reg_r}  ({reg_r/base-1:+.1%})")
    print(f"  Dense:          {dense_r}  ({dense_r/base-1:+.1%})")
    print()
    
    # Theory check
    print(f"  Hypothesis checks:")
    print(f"  'Comm helps'        {ring_r} < {base}? {'YES ✅' if ring_r < base else 'NO ❌'}")
    print(f"  'Better > worse'    {reg_r} < {ring_r}? {'YES ✅' if reg_r < ring_r else 'NO ❌'}")
    print(f"  'Dense is best'     {dense_r} <= {reg_r}? {'YES ✅' if dense_r <= reg_r else 'NO ❌'}")
    
    # Save results
    results_file = os.path.join(os.path.dirname(__file__) or '.', 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
