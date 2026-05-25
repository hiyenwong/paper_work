"""
Autonomous Research Loop for Topology Experiment

This script is called by Claude Code in an autoresearch loop:
1. Modify experiment hyperparameters
2. Run experiment
3. Check if topology effect is detected
4. Log results and iterate
"""

import json
import os
import subprocess
import sys
from datetime import datetime

LOG_FILE = os.path.join(os.path.dirname(__file__) or '.', 'autoresearch_log.json')
EXPERIMENT_SCRIPT = os.path.join(os.path.dirname(__file__) or '.', 'graph_signal_propagation.py')

def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            return json.load(f)
    return {"runs": [], "best_topology": None, "best_loss": float('inf')}

def save_log(log):
    with open(LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)

def run_experiment():
    """Run the experiment and return results."""
    result = subprocess.run(
        [sys.executable, '-u', EXPERIMENT_SCRIPT],
        capture_output=True, text=True, timeout=600
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500], file=sys.stderr)
    
    # Parse results from output
    lines = result.stdout.split('\n')
    results = {}
    for line in lines:
        if ':' in line and ('final_loss' in line.lower() or 'FINAL loss' in line):
            pass
    
    # Try to load results file
    res_file = os.path.join(os.path.dirname(__file__) or '.', 'smooth_results.json')
    if os.path.exists(res_file):
        with open(res_file) as f:
            return json.load(f)
    return None

def check_detection(results):
    """Check if topology effect is clearly detected."""
    if not results or len(results) < 3:
        return False
    
    by_name = {r['topology']: r for r in results}
    
    required = ['none', 'ring', 'random_regular', 'dense']
    if not all(n in by_name for n in required):
        return False
    
    baseline = by_name['none']['best_loss']
    ring_loss = by_name['ring']['best_loss']
    reg_loss = by_name['random_regular']['best_loss']
    dense_loss = by_name['dense']['best_loss']
    
    checks = {
        "random_regular_beats_none": reg_loss < baseline * 0.95,
        "random_regular_beats_ring": reg_loss < ring_loss * 0.95,
        "ring_worse_than_none": ring_loss > baseline,
        "effect_size_gt_10pct": reg_loss < baseline * 0.90,
    }
    
    return {
        "detected": all(checks.values()),
        "checks": checks,
        "values": {
            "none": baseline,
            "ring": ring_loss,
            "random_regular": reg_loss,
            "dense": dense_loss,
        }
    }

def suggest_improvements(results, detection):
    """Suggest next improvements based on results."""
    suggestions = []
    
    if not results or len(results) < 4:
        return ["Run all 4 topologies first"]
    
    by_name = {r['topology']: r for r in results}
    
    if detection and detection['detected']:
        # Already detected - try larger scale
        suggestions.append("TOPOLOGY EFFECT DETECTED! Now try:")
        suggestions.append("1. N=256, N=512 to see if gap widens")
        suggestions.append("2. Multiple seeds (5 runs each)")
        suggestions.append("3. Export results for paper")
    else:
        # Not detected yet - iterate
        reg_loss = by_name.get('random_regular', {}).get('best_loss', 1)
        none_loss = by_name.get('none', {}).get('best_loss', 0)
        
        if reg_loss > none_loss:
            suggestions.append("Random Regular not beating None:")
            suggestions.append("- Increase message passing steps (try 10, 16)")
            suggestions.append("- Increase N to 128, 256")
            suggestions.append("- Add skip connections in message passing")
        elif not detection['checks']['random_regular_beats_ring']:
            suggestions.append("Random Regular close but not clearly better:")
            suggestions.append("- Increase D to 64 for richer representations")
            suggestions.append("- Add multiple message passing layers")
            suggestions.append("- Try lower learning rate (1e-4)")
    
    return suggestions

def main():
    log = load_log()
    
    print(f"\n{'='*70}")
    print(f"AUTORESEARCH LOOP - Run #{len(log['runs']) + 1}")
    print(f"{'='*70}")
    
    print("\nRunning experiment...")
    results = run_experiment()
    
    if results:
        detection = check_detection(results)
        suggestions = suggest_improvements(results, detection)
        
        run_record = {
            "run_id": len(log['runs']) + 1,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "detection": detection,
            "suggestions": suggestions,
        }
        log['runs'].append(run_record)
        
        # Track best
        for r in results:
            if r.get('best_loss', 1) < log['best_loss']:
                log['best_loss'] = r['best_loss']
                log['best_topology'] = r['topology']
        
        save_log(log)
        
        print(f"\n{'='*70}")
        print("DETECTION STATUS:", end=" ")
        if detection and detection['detected']:
            print("✅ DETECTED!")
        else:
            print("❌ Not yet")
        print(f"{'='*70}")
        
        print(f"\nResults: {json.dumps({r['topology']: r['best_loss'] for r in results}, indent=2)}")
        print(f"\nNext suggestions:")
        for s in suggestions:
            print(f"  {s}")
    else:
        print("ERROR: Could not get results")

if __name__ == "__main__":
    main()
