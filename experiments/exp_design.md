# Experiment Design: Topology-Constrained Expert Communication

## Why previous experiments failed
Within-layer message passing on aggregated expert states is too weak.
The topology effect gets washed out because all expert outputs are already mixed
through the gating mechanism before the topology step.

## This experiment: Topology-Constrained Expert Routing

### Architecture
- N=64 experts, each with D=32 learned representation
- Experts communicate ONLY through the topology graph
- Each expert's output is computed from: its own input + aggregated neighbor states
- The graph topology determines how fast information spreads

### Key difference from Phase 2A
Phase 2A was synthetic (no learning). This version has:
- Learnable expert embeddings
- Learnable message passing weights
- A real training objective (reconstruct target signals)
- The model must LEARN to use the topology for effective propagation

### Why this should detect the topology effect
- Ring: information trickles slowly (O(N/d) diameter)
- Random Regular: information spreads fast (O(log N) diameter)
- Dense: instant spread
- The model's ability to learn depends on how fast information reaches across experts

### Training
- 500 steps, AdamW, lr=1e-3
- Measure: MSE between predicted and true signals
- Compare: none, ring, random_regular, dense
- Expected: dense < random_regular < ring < none

## Iteration Plan (autoresearch)
1. Initial implementation in PyTorch + MPS
2. Run and collect results
3. Identify issues and improve
4. Loop until topology effect is clearly detected
