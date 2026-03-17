#!/usr/bin/env python3
"""Run full pipeline on dataset and save results."""

import json
import argparse
from collections import defaultdict

from ontotrust_energy import build_memory, AtomicFeatureExtractor


def load_trajectories(path: str):
    """Load all trajectories from JSONL file."""
    trajectories = []
    with open(path, 'r') as f:
        for line in f:
            trajectories.append(json.loads(line))
    return trajectories


def run_pipeline(trajectory: dict, step_index: int):
    """Run pipeline for a single step."""
    memory = build_memory(trajectory, step_index=step_index)
    
    steps = trajectory.get('steps', [])
    if step_index <= len(steps):
        current_step = steps[step_index - 1]
        extractor = AtomicFeatureExtractor(use_llm_audit=False)
        features = extractor.extract(
            role_text=trajectory.get('role_text', ''),
            task_text=trajectory.get('task_text', ''),
            goal_text=current_step.get('goal_text', ''),
            evidence_text=current_step.get('evidence_text', ''),
            action_text=current_step.get('action_text', ''),
        )
        return {
            'trajectory_id': trajectory.get('trajectory_id', 'unknown'),
            'type': trajectory.get('type', 'unknown'),
            'step': step_index,
            'memory': memory,
            'features': features,
        }
    return None


def main():
    parser = argparse.ArgumentParser(description="Run pipeline on full dataset")
    parser.add_argument("--path", default="data/ontosync_v1_testset.jsonl")
    parser.add_argument("--output", default="data/pipeline_results.jsonl")
    args = parser.parse_args()
    
    # Load trajectories
    trajectories = load_trajectories(args.path)
    print(f"Loaded {len(trajectories)} trajectories")
    
    results = []
    feature_stats = defaultdict(int)
    type_stats = defaultdict(lambda: defaultdict(int))
    
    for traj in trajectories:
        traj_id = traj.get('trajectory_id', 'unknown')
        traj_type = traj.get('type', 'unknown')
        
        steps = traj.get('steps', [])
        num_steps = len(steps) + 1  # +1 for final step
        
        for step_idx in range(1, num_steps + 1):
            result = run_pipeline(traj, step_idx)
            if result:
                results.append(result)
                
                # Collect stats
                for feat_name, feat_value in result['features'].items():
                    if feat_value > 0:
                        feature_stats[feat_name] += 1
                        type_stats[traj_type][feat_name] += 1
    
    # Save results
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total results: {len(results)}")
    print(f"\nFeature counts (all trajectories):")
    for feat, count in sorted(feature_stats.items()):
        print(f"  {feat}: {count}")
    
    print(f"\nFeature counts by type:")
    for traj_type, feats in sorted(type_stats.items()):
        print(f"  {traj_type}:")
        for feat, count in sorted(feats.items()):
            print(f"    {feat}: {count}")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
