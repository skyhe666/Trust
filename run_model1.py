import json
from collections import defaultdict
from statistics import mean
from openai import OpenAI

from memory_builder import IncidentMemoryBuilder
from atomic_features import AtomicFeatureExtractor

DATA_PATH = "data/ontosync_v1_testset.jsonl"
MODEL = "/home/skyhe666/llm-model/Qwen/Qwen3.5-9B"

client = OpenAI(api_key="EMPTY", base_url="http://10.1.47.245:8000/v1/completions")


def llm_call(messages):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=400,
    )
    return resp.choices[0].message.content


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    trajectories = load_jsonl(DATA_PATH)

    memory_builder = IncidentMemoryBuilder()
    extractor = AtomicFeatureExtractor(llm_call=llm_call)

    all_step_results = []

    for traj in trajectories:
        traj_id = traj["trajectory_id"]
        collapse_type = traj["collapse_type"]
        label = traj["label"]

        print("=" * 80)
        print(f"Trajectory: {traj_id} | type={collapse_type} | label={label}")
        print("=" * 80)

        for step_idx, step in enumerate(traj["steps"]):
            memory_state = memory_builder.build(traj, step_idx).to_dict()
            result = extractor.extract_step(step, traj, memory_state).to_dict()

            row = {
                "trajectory_id": traj_id,
                "collapse_type": collapse_type,
                "label": label,
                "step_index": step_idx + 1,
                "step_label": step.get("step_label", 0),
                "action_text": step["action_text"],
                "memory_state": memory_state,
                **result,
            }
            all_step_results.append(row)

            print(f"\n--- Step {step_idx + 1} ---")
            print("Action:", step["action_text"])
            print("Memory:", json.dumps(memory_state, ensure_ascii=False))
            print("Atomic:", json.dumps({
                k: row[k] for k in [
                    "role_boundary_stretch",
                    "role_scope_mismatch",
                    "direct_task_divergence",
                    "incident_object_mismatch",
                    "unsupported_by_observed_evidence",
                    "evidence_gap",
                    "unanchored_target_reference",
                    "requires_scope_expansion",
                ]
            }, ensure_ascii=False))

    # 保存逐步结果
    with open("data/atomic_step_results.jsonl", "w", encoding="utf-8") as f:
        for row in all_step_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 按 collapse_type 聚合
    feature_names = [
        "role_boundary_stretch",
        "role_scope_mismatch",
        "direct_task_divergence",
        "incident_object_mismatch",
        "unsupported_by_observed_evidence",
        "evidence_gap",
        "unanchored_target_reference",
        "requires_scope_expansion",
    ]

    grouped = defaultdict(list)
    for row in all_step_results:
        grouped[row["collapse_type"]].append(row)

    print("\n" + "#" * 80)
    print("Per-collapse_type feature means")
    print("#" * 80)

    summary = {}
    for ctype, rows in grouped.items():
        summary[ctype] = {}
        print(f"\n[{ctype}]")
        for feat in feature_names:
            val = mean(r[feat] for r in rows) if rows else 0.0
            summary[ctype][feat] = val
            print(f"{feat:35s}: {val:.3f}")

    with open("data/atomic_feature_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print("- data/atomic_step_results.jsonl")
    print("- data/atomic_feature_summary.json")


if __name__ == "__main__":
    main()