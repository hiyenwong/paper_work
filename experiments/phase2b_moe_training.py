"""
Phase 2B: Tiny MoE Training Experiment with MLX

A minimal MoE transformer trained on tiny text data.
Tests how different expert communication topologies affect perplexity.

Architecture:
  - Vocab: 64 characters (char-level)
  - Embed: 64-dim
  - 1 Transformer layer with MoE
  - 8 experts, top-2 routing
  - Expert interaction: after MoE, experts exchange output through topology
  - Expert hidden dim: 64
  
Training: 500 steps, batch_size=32, seq_len=32
Dataset: synthetic text generated from a simple grammar
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import networkx as nx
import math
import time
from functools import partial


# ============================================================
# 1. Tiny Character-Level Tokenizer
# ============================================================
CHARS = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"-()\n"
VOCAB_SIZE = len(CHARS) + 4  # chars + <bos>, <eos>, <pad>, <unk>
char_to_id = {c: i + 4 for i, c in enumerate(CHARS)}
char_to_id['<bos>'] = 0
char_to_id['<eos>'] = 1
char_to_id['<pad>'] = 2
char_to_id['<unk>'] = 3
id_to_char = {v: k for k, v in char_to_id.items()}


def encode(text: str) -> list:
    return [char_to_id.get(c, 3) for c in text]


def decode(ids) -> str:
    if isinstance(ids, mx.array):
        ids = ids.tolist()
    return ''.join(id_to_char.get(i, '?') for i in ids)


# ============================================================
# 2. Tiny Training Dataset
# ============================================================
TRAIN_TEXT = """
the quick brown fox jumps over the lazy dog
the five boxing wizards jump quickly
the cat sat on the mat and the dog ran after it
the bird flew high above the trees
the fish swam deep in the blue ocean
hello world this is a test of the language model
machine learning is fun and interesting
the expert model learns to route tokens efficiently
sparse mixture of experts uses topk routing
the ramanujan graph has optimal spectral gap
the expander graph mixes information fast
the ring graph needs many hops to cover all nodes
the random regular graph has logarithmic diameter
the dense graph connects all experts directly
a good topology propagates information quickly
the neural network learns patterns from data
training requires many steps and careful tuning
the loss function measures prediction error
gradient descent updates the model weights
backpropagation computes gradients efficiently
the transformer uses self attention mechanisms
multi head attention captures different relationships
the embedding layer converts tokens to vectors
the positional encoding adds order information
the output layer predicts the next token
the cross entropy loss works well for classification
the softmax function produces probability distributions
the adam optimizer adapts learning rates
the learning rate schedule affects convergence
batch normalization stabilizes training
dropout prevents overfitting in neural networks
the validation set helps tune hyperparameters
the test set evaluates final model performance
data augmentation improves model generalization
ensemble methods combine multiple models
transfer learning adapts pretrained models
fine tuning specializes models for specific tasks
knowledge distillation compresses large models
quantization reduces model size for deployment
pruning removes unnecessary connections
the inference speed matters for production
"""


def generate_batch(text: str, seq_len: int = 32, batch_size: int = 32) -> tuple:
    """Generate a batch of (input, target) sequences."""
    tokens = encode(text)
    L = len(tokens)
    inputs = []
    targets = []
    np.random.seed(int(time.time()) % 10000)
    for _ in range(batch_size):
        start = np.random.randint(0, max(1, L - seq_len - 1))
        inp = tokens[start:start + seq_len]
        tgt = tokens[start + 1:start + seq_len + 1]
        if len(inp) < seq_len:
            inp = inp + [2] * (seq_len - len(inp))
            tgt = tgt + [2] * (seq_len - len(tgt))
        inputs.append(inp)
        targets.append(tgt)
    return mx.array(inputs), mx.array(targets)


# ============================================================
# 3. Graph Topology Builder
# ============================================================
def build_graph(name: str, N: int, d: int, seed: int = 42) -> tuple:
    """Build graph and return (adjacency_list, is_connected)."""
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
        raise ValueError(f"Unknown topology: {name}")
    
    adj = {i: list(G.neighbors(i)) for i in range(N)}
    return adj, nx.is_connected(G)


def make_message_matrix(adj_list: dict, d: int, N: int) -> mx.array:
    """Create a message aggregation matrix.
    
    Each expert sends its output to its neighbors and receives from neighbors.
    P[i,j] = 1/(d+1) if j is neighbor of i or j=i.
    """
    # Build matrix row by row
    rows = []
    for i in range(N):
        row = mx.zeros(N)
        row = row.at[i].add(1.0 / (d + 1))  # self-loop
        for j in adj_list.get(i, []):
            row = row.at[j].add(1.0 / (d + 1))
        rows.append(row)
    return mx.stack(rows)


# ============================================================
# 4. Tiny MoE Transformer
# ============================================================
class Expert(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )
    
    def __call__(self, x: mx.array) -> mx.array:
        return self.net(x)


class TinyMoELayer(nn.Module):
    def __init__(self, dim: int, num_experts: int, top_k: int, 
                 topology_name: str, expert_degree: int):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_active = top_k
        self.topology = topology_name
        
        # Experts
        self.experts = [Expert(dim) for _ in range(num_experts)]
        
        # Gating
        self.gate = nn.Linear(dim, num_experts)
        
        # Topology message matrix
        if topology_name != "none":
            adj_list, connected = build_graph(topology_name, num_experts, expert_degree)
            self.msg_matrix = make_message_matrix(adj_list, expert_degree, num_experts)
            self._connected = connected
        else:
            self.msg_matrix = None
            self._connected = True
        
        # Output projection
        self.output_proj = nn.Linear(dim, dim)
    
    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, seq, dim)
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)  # (N, D)
        N = B * S
        
        # Gating: softmax over experts
        gate_logits = self.gate(x_flat)  # (N, E)
        gate_weights = mx.softmax(gate_logits, axis=-1)
        
        # Top-k: make routing discrete (detach from gradients)
        gw_detached = mx.stop_gradient(gate_weights)
        sorted_inds = mx.argsort(gw_detached, axis=-1)  # ascending
        gate_mask = mx.zeros_like(gw_detached)
        gate_mask = gate_mask.at[:, sorted_inds[:, -self.num_active:]].add(1.0)
        gate_masked = gate_weights * gate_mask
        gate_masked = gate_masked / (mx.sum(gate_masked, axis=-1, keepdims=True) + 1e-8)
        
        # All experts process all tokens (no scatter!)
        # Using a list since mx.stack may fail; we will use a loop
        expert_out_list = []
        for e_idx in range(self.num_experts):
            e_out = self.experts[e_idx](x_flat)  # (N, D)
            expert_out_list.append(e_out)
        # (E, N, D)
        expert_out_stacked = mx.stack(expert_out_list)
        
        # Apply gate weights: (E, N, 1) * (E, N, D) -> (E, N, D)
        weights = mx.stop_gradient(gate_masked.T[:, :, None])  # (E, N, 1)
        weighted = expert_out_stacked * weights  # (E, N, D)
        
        # Expert communication via topology
        if self.topology != "none":
            # (E, N, D) -> (N, D, E) @ (E, E) -> (N, D, E) -> (E, N, D)
            comm_input = weighted.transpose(1, 2, 0)  # (N, D, E)
            comm_output = comm_input @ self.msg_matrix.T  # (N, D, E)
            aggregated = comm_output.transpose(2, 0, 1)  # (E, N, D)
        else:
            aggregated = weighted
        
        final_output = mx.sum(aggregated, axis=0)  # (N, D)
        return self.output_proj(final_output.reshape(B, S, D))


class TinyMoETransformer(nn.Module):
    def __init__(self, vocab_size: int, dim: int, num_heads: int,
                 num_experts: int, top_k: int, 
                 topology_name: str = "random_regular",
                 expert_degree: int = 3):
        super().__init__()
        self.dim = dim
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(512, dim)
        
        # Self-attention
        self.attention = nn.MultiHeadAttention(dim, num_heads)
        self.norm1 = nn.RMSNorm(dim)
        
        # MoE
        self.moe = TinyMoELayer(dim, num_experts, top_k, topology_name, expert_degree)
        self.norm2 = nn.RMSNorm(dim)
        
        # Output
        self.out_norm = nn.RMSNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size)
    
    def __call__(self, x: mx.array) -> mx.array:
        B, S = x.shape
        
        # Embeddings
        tok = self.token_embed(x)
        pos = self.pos_embed(mx.arange(S)[None, :])
        h = tok + pos
        
        # Self-attention
        attn_out = self.attention(self.norm1(h), self.norm1(h), self.norm1(h))
        h = h + attn_out
        
        # MoE
        moe_out = self.moe(self.norm2(h))
        h = h + moe_out
        
        # Output
        h = self.out_norm(h)
        logits = self.out_proj(h)
        return logits


# ============================================================
# 5. Training Loop
# ============================================================
def loss_fn(model, inputs, targets):
    logits = model(inputs)
    B, S, V = logits.shape
    logits = logits.reshape(-1, V)
    targets = targets.reshape(-1)
    loss = nn.losses.cross_entropy(logits, targets, reduction='mean')
    return loss


def train_one_batch(model, optimizer, inputs, targets):
    loss_fn_vg = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_fn_vg(model, inputs, targets)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return loss


def compute_perplexity(logits, targets):
    B, S, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    
    # Softmax + log
    probs = mx.softmax(logits_flat, axis=-1)
    target_probs = probs[mx.arange(len(targets_flat)), targets_flat]
    
    # Avoid log(0)
    eps = 1e-10
    log_probs = mx.log(mx.maximum(target_probs, eps))
    nll = -mx.mean(log_probs)
    ppl = mx.exp(nll)
    return ppl


def evaluate(model, text: str, seq_len: int = 32, num_samples: int = 10):
    """Evaluate on multiple random segments for stable perplexity."""
    tokens = encode(text)
    if len(tokens) < seq_len + 1:
        return None
    
    np.random.seed(0)
    nlls = []
    for _ in range(num_samples):
        start = np.random.randint(0, len(tokens) - seq_len - 1)
        inp = mx.array([tokens[start:start + seq_len]])
        tgt = mx.array([tokens[start + 1:start + seq_len + 1]])
        
        logits = model(inp)
        B, S, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = tgt.reshape(-1)
        
        loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='mean')
        nlls.append(loss.item())
    
    avg_nll = sum(nlls) / len(nlls)
    return math.exp(avg_nll)


# ============================================================
# 6. Run Experiment
# ============================================================
def run_experiment(topology_name: str, num_steps: int = 300, 
                   dim: int = 64, num_heads: int = 4,
                   num_experts: int = 8, top_k: int = 2,
                   expert_degree: int = 3,
                   lr: float = 3e-4, seed: int = 42) -> dict:
    """Run one topology experiment and return training metrics."""
    
    mx.random.seed(seed)
    np.random.seed(seed)
    
    # Model
    model = TinyMoETransformer(
        vocab_size=VOCAB_SIZE,
        dim=dim,
        num_heads=num_heads,
        num_experts=num_experts,
        top_k=top_k,
        topology_name=topology_name,
        expert_degree=expert_degree,
    )
    mx.eval(model.parameters())
    
    # Optimizer
    optimizer = optim.Adam(learning_rate=lr)
    
    # Metrics
    losses = []
    ppls = []
    
    start_time = time.time()
    print(f"\n  Training {topology_name} (N={num_experts}, d={expert_degree})...")
    
    for step in range(num_steps):
        inputs, targets = generate_batch(TRAIN_TEXT, seq_len=32, batch_size=16)
        loss = train_one_batch(model, optimizer, inputs, targets)
        
        if step % 50 == 0:
            ppl = evaluate(model, TRAIN_TEXT)
            losses.append(loss.item())
            ppls.append(ppl if ppl else 0)
            elapsed = time.time() - start_time
            print(f"    Step {step:3d}: loss={loss.item():.4f}, ppl={ppl:.2f} ({elapsed:.0f}s)")
    
    # Final evaluation
    final_ppl = evaluate(model, TRAIN_TEXT)
    elapsed = time.time() - start_time
    
    print(f"  {topology_name} done: final ppl={final_ppl:.2f}, time={elapsed:.0f}s")
    
    return {
        "topology": topology_name,
        "final_loss": float(loss.item()),
        "final_ppl": float(final_ppl) if final_ppl else 0,
        "time_sec": elapsed,
        "losses": losses,
        "ppls": ppls,
    }


def main():
    print("=" * 80)
    print("PHASE 2B: TINY MoE TRAINING EXPERIMENT (MLX)")
    print("=" * 80)
    
    # Configuration
    NB_EXPERTS = 64
    TOP_K = 4       # top-4 for larger expert pool
    EXPERT_DEGREE = 4
    NUM_STEPS = 200
    DIM = 32        # reduce dim to keep computation feasible
    N_HEADS = 4
    
    print(f"\nConfiguration:")
    print(f"  Architecture: 1-layer Transformer + MoE")
    print(f"  Embed dim: {DIM}, Heads: {N_HEADS}")
    print(f"  Num experts: {NB_EXPERTS}, Top-K: {TOP_K}, Degree: {EXPERT_DEGREE}")
    print(f"  Vocab: {VOCAB_SIZE} chars, Training steps: {NUM_STEPS}")
    print(f"  Dataset: {len(TRAIN_TEXT)} chars")
    
    topologies = [
        "none",       # No expert communication (baseline)
        "ring",       # Ring topology (poor expander)
        "random_regular",  # Near-Ramanujan expander
        "dense",      # Dense (upper bound)
    ]
    
    results = []
    for topo in topologies:
        r = run_experiment(
            topology_name=topo,
            num_steps=NUM_STEPS,
            dim=DIM,
            num_heads=N_HEADS,
            num_experts=NB_EXPERTS,
            top_k=TOP_K,
            expert_degree=EXPERT_DEGREE,
            seed=42,
        )
        results.append(r)
    
    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Topology':<20} {'Final PPL':<12} {'Final Loss':<12} {'Time (s)':<10}")
    print("-" * 54)
    for r in results:
        print(f"{r['topology']:<20} {r['final_ppl']:<12.2f} {r['final_loss']:<12.4f} {r['time_sec']:<10.0f}")
    print("-" * 54)
    print(f"\nVerdict: If random_regular achieves comparable perplexity to 'none'\n"
          f"(which has no communication cost), the topology provides free\n"
          f"information mixing at no perplexity cost. If it outperforms ring,\n"
          f"the expansion property directly benefits MoE training.\n")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
