# Topology Experiment - Autoresearch

## Goal
Demonstrate that expander graph topology (Random Regular) enables significantly better information propagation in MoE-like expert communication compared to Ring and None topologies.

## Current Best Result
Random Regular loss=0.02384 vs None loss=0.03526 (-32%) vs Ring loss=0.06627 (-64%)

## Experiment Scripts
- `graph_signal_propagation.py` — Main experiment (PyTorch + MPS)
- `autoresearch_loop.py` — Autonomous iteration loop

## Key Parameters (graph_signal_propagation.py)
- N=64 experts, D=32 dim, degree=4
- 8 message passing steps
- Shared targets generated from random_regular graph
- Measure: final/best MSE loss

## Iteration Plan
1. Run `python3 -u graph_signal_propagation.py` to get baseline
2. Increase N (128, 256) to see gap widen
3. Try different degrees (6, 8)
4. Run with multiple seeds (42, 123, 456) for statistical significance
5. Generate summary table for paper
6. Commit results to git

## Rules
- Always compare: none, ring, random_regular, dense
- Use shared targets (generate once from random_regular)
- Save results to smooth_results.json
- Log each run in autoresearch_log.json
- Never delete existing results files
- Use `git add -A && git commit -m "..." && git push origin main` after significant findings
