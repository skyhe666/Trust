#!/usr/bin/env python3
"""Test script for Incident Memory Builder with LLM."""

import argparse
import json
from pathlib import Path

from ontotrust_energy import IncidentMemoryBuilder


def load_trajectories(path: str, limit: int = None):
    """Load trajectories from JSONL file."""
    trajectories = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
                if limit and len(trajectories) >= limit:
                    break
    return trajectories


def main():
    parser = argparse.ArgumentParser(description="Test Incident Memory Builder")
    parser.add_argument(
        '--data',
        type=str,
        default='data/ontosync_v1_testset.jsonl',
        help='Path to JSONL trajectory file'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=2,
        help='Number of trajectories to test'
    )
    parser.add_argument(
        '--step',
        type=int,
        default=None,
        help='Test specific step (1-based), default=all steps'
    )
    args = parser.parse_args()
    
    # Resolve path
    data_path = Path(args.data)
    if not data_path.is_absolute():
        project_root = Path(__file__).parent.parent
        data_path = project_root / data_path
    
    print(f"Loading trajectories from: {data_path}")
    trajectories = load_trajectories(str(data_path), limit=args.limit)
    print(f"Loaded {len(trajectories)} trajectories\n")
    
    # Create builder
    builder = IncidentMemoryBuilder()
    
    for traj in trajectories:
        traj_id = traj.get('trajectory_id', 'unknown')
        collapse_type = traj.get('collapse_type', 'unknown')
        
        print("=" * 80)
        print(f"Trajectory: {traj_id} (type: {collapse_type})")
        print("=" * 80)
        
        if args.step:
            # Test specific step
            print(f"\n--- Memory for step {args.step} (steps 1 to {args.step-1}) ---")
            memory = builder.build_memory(traj, step_index=args.step)
            print(json.dumps(memory, indent=2, ensure_ascii=False))
        else:
            # Test all steps
            steps = traj.get('steps', [])
            print(f"Total steps: {len(steps)}\n")
            
            for i in range(1, len(steps) + 1):
                print(f"--- Step {i} memory (from steps 1 to {i-1}) ---")
                memory = builder.build_memory(traj, step_index=i)
                
                # Pretty print
                print(f"  primary_object: {memory['primary_object']}")
                print(f"  primary_object_type: {memory['primary_object_type']}")
                print(f"  core_task: {memory['core_task'][:60]}...")
                print(f"  key_findings: {memory['key_findings'][:2]}")
                print(f"  action_history: {memory['action_history'][:2]}")
                print(f"  has_object_switch: {memory['has_object_switch']}")
                print(f"  has_goal_shift: {memory['has_goal_shift']}")
                print(f"  evidence_sufficient: {memory['evidence_sufficient']}")
                print(f"  scope_expanded: {memory['scope_expanded']}")
                print(f"  steps_summarized: {memory['steps_summarized']}")
                print()
        
        print()


if __name__ == '__main__':
    main()
