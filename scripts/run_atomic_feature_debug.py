#!/usr/bin/env python3
"""Debug script for atomic feature extraction.

Reads JSONL trajectory files and outputs atomic features for each step,
plus aggregated statistics by collapse_type.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ontotrust_energy import AtomicFeatureExtractor, ATOMIC_FEATURE_NAMES


def load_trajectories(path: str) -> list:
    """Load trajectories from JSONL file."""
    trajectories = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
    return trajectories


def extract_all_features(trajectories: list) -> list:
    """Extract features for all steps in all trajectories."""
    extractor = AtomicFeatureExtractor()
    results = []
    
    for traj in trajectories:
        traj_id = traj.get('trajectory_id', 'unknown')
        collapse_type = traj.get('collapse_type', 'unknown')
        task_text = traj.get('task_text', '')
        role_text = traj.get('role_text', '')
        
        steps = traj.get('steps', [])
        for step in steps:
            step_index = step.get('step_index', 0)
            action_text = step.get('action_text', '')
            goal_text = step.get('goal_text', '')
            evidence_text = step.get('evidence_text', '')
            step_label = step.get('step_label', None)
            
            # Extract features
            features = extractor.extract(
                role_text=role_text,
                task_text=task_text,
                goal_text=goal_text,
                evidence_text=evidence_text,
                action_text=action_text,
            )
            
            results.append({
                'trajectory_id': traj_id,
                'collapse_type': collapse_type,
                'step_index': step_index,
                'step_label': step_label,
                **features,
            })
    
    return results


def print_step_results(results: list):
    """Print individual step results."""
    print("\n" + "=" * 120)
    print("INDIVIDUAL STEP RESULTS")
    print("=" * 120)
    
    # Print header
    header = f"{'trajectory_id':<30} {'collapse_type':<20} {'step':<6} {'label':<6} "
    header += " ".join([f"{name[:4]:>4}" for name in ATOMIC_FEATURE_NAMES])
    print(header)
    print("-" * 120)
    
    for r in results:
        row = f"{r['trajectory_id']:<30} {r['collapse_type']:<20} {r['step_index']:<6} "
        row += f"{r['step_label'] if r['step_label'] is not None else '-':<6} "
        row += " ".join([f"{r[feat]:>4}" for feat in ATOMIC_FEATURE_NAMES])
        print(row)


def aggregate_by_collapse_type(results: list) -> dict:
    """Aggregate features by collapse_type."""
    agg = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        ct = r['collapse_type']
        for feat in ATOMIC_FEATURE_NAMES:
            agg[ct][feat].append(r[feat])
    
    # Compute averages
    averages = {}
    for ct, features in agg.items():
        averages[ct] = {}
        for feat, values in features.items():
            averages[ct][feat] = sum(values) / len(values) if values else 0
    
    return averages


def print_aggregated_results(averages: dict):
    """Print aggregated results by collapse_type."""
    print("\n" + "=" * 120)
    print("AGGREGATED BY COLLAPSE TYPE (Mean Feature Values)")
    print("=" * 120)
    
    # Get all collapse types
    collapse_types = sorted(averages.keys())
    
    # Print header
    header = f"{'collapse_type':<20} "
    header += " ".join([f"{name[:4]:>8}" for name in ATOMIC_FEATURE_NAMES])
    print(header)
    print("-" * 120)
    
    for ct in collapse_types:
        row = f"{ct:<20} "
        row += " ".join([f"{averages[ct][feat]:>8.3f}" for feat in ATOMIC_FEATURE_NAMES])
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Debug atomic feature extraction on trajectory data."
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/ontosync_v1_testset.jsonl',
        help='Path to JSONL trajectory file'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Limit to first N trajectories'
    )
    args = parser.parse_args()
    
    # Resolve data path
    data_path = Path(args.data)
    if not data_path.is_absolute():
        # Assume relative to project root
        project_root = Path(__file__).parent.parent
        data_path = project_root / data_path
    
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return
    
    print(f"Loading trajectories from: {data_path}")
    trajectories = load_trajectories(str(data_path))
    
    if args.sample:
        trajectories = trajectories[:args.sample]
    
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Extract features
    results = extract_all_features(trajectories)
    print(f"Extracted features for {len(results)} steps")
    
    # Print individual results
    print_step_results(results)
    
    # Aggregate and print
    averages = aggregate_by_collapse_type(results)
    print_aggregated_results(averages)
    
    # Summary stats
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    for ct in sorted(averages.keys()):
        total = sum(1 for r in results if r['collapse_type'] == ct)
        avg_total = sum(averages[ct].values()) / len(ATOMIC_FEATURE_NAMES)
        print(f"  {ct}: {total} steps, avg total features = {avg_total:.3f}")


if __name__ == '__main__':
    main()
