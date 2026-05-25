"""
run_all_experiments.py
======================
Systematic sweep over:
  - N: 64, 128, 256
  - degree: 4, 6, 8
  - seeds: 42, 123, 456
  - topologies: none, ring, random_regular, dense

Saves to all_results.json and prints a summary table.
"""

import json
import os
import sys
import time
import numpy as np

# Import core helpers from the existing module
sys.path.insert(0, os.path.dirname(__file__))
from graph_signal_propagation import (
    run_experiment,
    generate_smooth_targets,
)

TOPOLOGIES = ["none", "ring", "random_regular", "dense"]
SEEDS = [42, 123, 456]
NS = [64, 128, 256]
DEGREES = [4, 6, 8]
STEPS = 500

RESULTS_FILE = os.path.join(os.path.dirname(__file__) or ".", "all_results.json")


def load_existing():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def result_key(r):
    return (r["N"], r["degree"], r["seed"], r["topology"])


def already_done(existing, N, degree, seed, topology):
    keys = {result_key(r) for r in existing}
    return (N, degree, seed, topology) in keys


def print_table(results, label=""):
    """Print a grouped summary table."""
    if not results:
        return
    print(f"\n{'='*80}")
    if label:
        print(f"  {label}")
    print(f"{'='*80}")
    header = f"{'N':>5} {'deg':>4} {'seed':>5} {'Topology':<16} {'Best Loss':>10} {'vs None':>8}"
    print(header)
    print("-" * 50)

    # Group by (N, degree, seed)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[(r["N"], r["degree"], r["seed"])].append(r)

    for (N, deg, seed), runs in sorted(groups.items()):
        none_r = next((r for r in runs if r["topology"] == "none"), None)
        base = none_r["best_loss"] if none_r else None
        for r in sorted(runs, key=lambda x: TOPOLOGIES.index(x["topology"])):
            vs = f"{r['best_loss']/base-1:+.1%}" if base else "N/A"
            mark = " *" if r["topology"] == "random_regular" else ""
            print(f"{N:>5} {deg:>4} {seed:>5} {r['topology']:<16} {r['best_loss']:>10.6f} {vs:>8}{mark}")
        print()


def sweep_seeds(N, degree, label):
    """Run all topologies across 3 seeds for given N, degree."""
    existing = load_existing()
    new_results = []

    for seed in SEEDS:
        # Generate shared targets once per (N, degree, seed)
        np.random.seed(seed)
        shared_targets = generate_smooth_targets(N, 32, "random_regular", degree)

        for topo in TOPOLOGIES:
            if already_done(existing, N, degree, seed, topo):
                print(f"  [SKIP] N={N} deg={degree} seed={seed} {topo} (cached)")
                continue

            print(f"\n--- N={N} deg={degree} seed={seed} {topo} ---", flush=True)
            r = run_experiment(
                topo,
                N=N,
                D=32,
                shared_targets=shared_targets,
                degree=degree,
                steps=STEPS,
                lr=3e-4,
                seed=seed,
            )
            r["N"] = N
            r["degree"] = degree
            r["seed"] = seed
            existing.append(r)
            new_results.append(r)
            save_results(existing)
            print(f"  Saved. {len(existing)} total results.", flush=True)

    return existing


def main():
    total_start = time.time()

    print("=" * 80)
    print("COMPREHENSIVE TOPOLOGY SWEEP")
    print(f"  N: {NS}")
    print(f"  degrees: {DEGREES}")
    print(f"  seeds: {SEEDS}")
    print(f"  topologies: {TOPOLOGIES}")
    print(f"  steps per run: {STEPS}")
    print("=" * 80)

    # 1. Seed sweep: N=64, degree=4 (baseline)
    print("\n[1/5] Baseline: N=64, degree=4, seeds=42,123,456")
    all_results = sweep_seeds(N=64, degree=4, label="baseline")

    # 2. Scale N: N=128, degree=4, seeds=42,123,456
    print("\n[2/5] Scale N: N=128, degree=4")
    all_results = sweep_seeds(N=128, degree=4, label="N=128")

    # 3. Scale N: N=256, degree=4, seeds=42,123,456
    print("\n[3/5] Scale N: N=256, degree=4")
    all_results = sweep_seeds(N=256, degree=4, label="N=256")

    # 4. Degree sweep: N=64, degrees=6,8, seed=42
    for deg in [6, 8]:
        print(f"\n[4/5] Degree sweep: N=64, degree={deg}, seeds=42,123,456")
        all_results = sweep_seeds(N=64, degree=deg, label=f"N=64 deg={deg}")

    # 5. Summary
    print_table(all_results, label="ALL RESULTS")

    # Aggregate: mean best_loss per (N, degree, topology) across seeds
    print("\n" + "=" * 80)
    print("  AGGREGATED (mean ± std over seeds)")
    print("=" * 80)
    from collections import defaultdict
    agg = defaultdict(list)
    for r in all_results:
        agg[(r["N"], r["degree"], r["topology"])].append(r["best_loss"])

    print(f"{'N':>5} {'deg':>4} {'Topology':<16} {'Mean Loss':>10} {'Std':>8} {'vs None':>8}")
    print("-" * 55)

    for (N, deg) in sorted({(r["N"], r["degree"]) for r in all_results}):
        none_vals = agg.get((N, deg, "none"), [])
        base_mean = np.mean(none_vals) if none_vals else None
        for topo in TOPOLOGIES:
            vals = agg.get((N, deg, topo), [])
            if not vals:
                continue
            mean = np.mean(vals)
            std = np.std(vals)
            vs = f"{mean/base_mean-1:+.1%}" if base_mean else "N/A"
            mark = " *" if topo == "random_regular" else ""
            print(f"{N:>5} {deg:>4} {topo:<16} {mean:>10.6f} {std:>8.6f} {vs:>8}{mark}")
        print()

    # Key checks
    print("\n" + "=" * 80)
    print("  KEY HYPOTHESIS CHECKS (mean over seeds)")
    print("=" * 80)
    checks_passed = 0
    checks_total = 0
    for (N, deg) in sorted({(r["N"], r["degree"]) for r in all_results}):
        none_mean = np.mean(agg.get((N, deg, "none"), [np.nan]))
        ring_mean = np.mean(agg.get((N, deg, "ring"), [np.nan]))
        rr_mean = np.mean(agg.get((N, deg, "random_regular"), [np.nan]))
        dense_mean = np.mean(agg.get((N, deg, "dense"), [np.nan]))

        c1 = rr_mean < none_mean
        c2 = rr_mean < ring_mean
        checks = [
            (f"N={N} deg={deg}: RR < None", c1, f"{rr_mean:.4f} vs {none_mean:.4f}"),
            (f"N={N} deg={deg}: RR < Ring", c2, f"{rr_mean:.4f} vs {ring_mean:.4f}"),
        ]
        for desc, ok, detail in checks:
            checks_total += 1
            if ok:
                checks_passed += 1
            icon = "PASS" if ok else "FAIL"
            print(f"  [{icon}] {desc} ({detail})")

    print(f"\n  Score: {checks_passed}/{checks_total} checks passed")

    elapsed = time.time() - total_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"  Results saved to {RESULTS_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
