import json
from collections import defaultdict
from statistics import mean, pstdev
from openai import OpenAI

from memory_builder import IncidentMemoryBuilder
from atomic_features import AtomicFeatureExtractor

DATA_PATH = "data/ontosync_v1_testset.jsonl"
MODEL = "/home/skyhe666/llm-model/Qwen/Qwen3.5-9B"

client = OpenAI(api_key="EMPTY", base_url="http://10.1.47.245:8000/v1")


def llm_call(messages):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=2048,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    msg = resp.choices[0].message
    return (msg.content or "").strip()


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_mean(xs):
    return mean(xs) if xs else 0.0


def safe_std(xs):
    return pstdev(xs) if len(xs) > 1 else 0.0


def main():
    trajectories = load_jsonl(DATA_PATH)

    memory_builder = IncidentMemoryBuilder(llm_call=llm_call)
    extractor = AtomicFeatureExtractor(llm_call=llm_call)

    all_step_results = []

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

    # ========== 逐步运行 ==========
    for traj in trajectories:
        traj_id = traj["trajectory_id"]
        collapse_type = traj["collapse_type"]
        label = traj["label"]
        deviation_step = traj.get("deviation_step", None)

        print("=" * 80)
        print(f"Trajectory: {traj_id} | type={collapse_type} | label={label}")
        print("=" * 80)

        for step_idx, step in enumerate(traj["steps"]):
            memory_obj = memory_builder.build(traj, step_idx)
            memory_state = memory_obj.to_dict()

            result = extractor.extract_step(step, traj, memory_state).to_dict()

            atomic_values = {k: result[k] for k in feature_names}
            active_count = sum(atomic_values.values())

            row = {
                "trajectory_id": traj_id,
                "collapse_type": collapse_type,
                "label": label,
                "deviation_step": deviation_step,
                "step_index": step_idx + 1,
                "step_label": step.get("step_label", 0),
                "action_text": step.get("action_text", ""),
                "goal_text": step.get("goal_text", ""),
                "evidence_text": step.get("evidence_text", ""),
                "memory_state": memory_state,
                "active_feature_count": active_count,
                **result,
            }
            all_step_results.append(row)

            print(f"\n--- Step {step_idx + 1} ---")
            print("Action:", step.get("action_text", ""))
            print("Memory core:", memory_state.get("core_incident_objects", []))
            print("Memory incident-anchored:", memory_state.get("incident_anchored_objects", []))
            print("Memory historically-seen:", memory_state.get("historically_seen_objects", []))
            print("Memory findings:", memory_state.get("observed_findings", [])[:5])
            print("Atomic:", json.dumps(atomic_values, ensure_ascii=False))
            print("Active feature count:", active_count)

            if (
                not memory_state.get("core_incident_objects")
                and memory_state.get("raw_core_response")
            ):
                print("MEMORY CORE RAW:", memory_state["raw_core_response"][:800])

            if (
                step_idx > 0
                and not memory_state.get("incident_anchored_objects")
                and not memory_state.get("historically_seen_objects")
                and not memory_state.get("observed_findings")
                and memory_state.get("raw_history_response")
            ):
                print("MEMORY HISTORY RAW:", memory_state["raw_history_response"][:800])

            if sum(atomic_values.values()) == 0 and row.get("raw_response"):
                print("ATOMIC RAW RESPONSE:", row["raw_response"][:800])

    # 保存逐步结果
    with open("data/atomic_step_results.jsonl", "w", encoding="utf-8") as f:
        for row in all_step_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ========== 统计 1：按 collapse_type（benign 全部；malicious 只看 post-deviation） ==========
    grouped = defaultdict(list)
    for row in all_step_results:
        if row["collapse_type"] == "benign":
            grouped["benign"].append(row)
        elif row["step_label"] == 1:
            grouped[row["collapse_type"]].append(row)

    print("\n" + "#" * 80)
    print("Per-collapse_type feature means (benign=all, malicious=post-deviation only)")
    print("#" * 80)

    summary = {}
    for ctype, rows in grouped.items():
        summary[ctype] = {}
        print(f"\n[{ctype}]  n={len(rows)}")
        for feat in feature_names:
            vals = [r[feat] for r in rows]
            val = safe_mean(vals)
            summary[ctype][feat] = val
            print(f"{feat:35s}: {val:.3f}")

    with open("data/atomic_feature_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ========== 统计 2：benign vs malicious_post ==========
    binary_grouped = {"benign": [], "malicious_post": []}
    for row in all_step_results:
        if row["collapse_type"] == "benign":
            binary_grouped["benign"].append(row)
        elif row["step_label"] == 1:
            binary_grouped["malicious_post"].append(row)

    print("\n" + "#" * 80)
    print("Benign vs Malicious-Post feature means")
    print("#" * 80)

    binary_summary = {}
    for gname, rows in binary_grouped.items():
        binary_summary[gname] = {}
        print(f"\n[{gname}]  n={len(rows)}")
        for feat in feature_names:
            vals = [r[feat] for r in rows]
            val = safe_mean(vals)
            binary_summary[gname][feat] = val
            print(f"{feat:35s}: {val:.3f}")

    with open("data/atomic_feature_binary_summary.json", "w", encoding="utf-8") as f:
        json.dump(binary_summary, f, ensure_ascii=False, indent=2)

    # ========== 统计 3：特征分布统计（mean/std/activation_rate） ==========
    print("\n" + "#" * 80)
    print("Feature distribution stats")
    print("#" * 80)

    distribution_stats = {}
    for gname, rows in binary_grouped.items():
        distribution_stats[gname] = {}
        print(f"\n[{gname}]  n={len(rows)}")
        for feat in feature_names:
            vals = [r[feat] for r in rows]
            stat = {
                "mean": safe_mean(vals),
                "std": safe_std(vals),
                "activation_rate": safe_mean(vals),  # 二值特征下和 mean 相同
            }
            distribution_stats[gname][feat] = stat
            print(
                f"{feat:35s}: mean={stat['mean']:.3f}  std={stat['std']:.3f}  act_rate={stat['activation_rate']:.3f}"
            )

    with open("data/atomic_feature_distribution_stats.json", "w", encoding="utf-8") as f:
        json.dump(distribution_stats, f, ensure_ascii=False, indent=2)

    # ========== 统计 4：特征饱和度（每步亮几个特征） ==========
    print("\n" + "#" * 80)
    print("Feature saturation stats")
    print("#" * 80)

    saturation_stats = {}
    for gname, rows in binary_grouped.items():
        counts = [r["active_feature_count"] for r in rows]
        saturation_stats[gname] = {
            "mean_active_count": safe_mean(counts),
            "std_active_count": safe_std(counts),
            "zero_feature_rate": safe_mean([1 if c == 0 else 0 for c in counts]),
            "full_feature_rate": safe_mean([1 if c == len(feature_names) else 0 for c in counts]),
        }

        print(f"\n[{gname}]  n={len(rows)}")
        print(f"mean_active_count : {saturation_stats[gname]['mean_active_count']:.3f}")
        print(f"std_active_count  : {saturation_stats[gname]['std_active_count']:.3f}")
        print(f"zero_feature_rate : {saturation_stats[gname]['zero_feature_rate']:.3f}")
        print(f"full_feature_rate : {saturation_stats[gname]['full_feature_rate']:.3f}")

    with open("data/atomic_feature_saturation_stats.json", "w", encoding="utf-8") as f:
        json.dump(saturation_stats, f, ensure_ascii=False, indent=2)

    # ========== 统计 5：起漂步统计 ==========
    print("\n" + "#" * 80)
    print("First-drift-step stats")
    print("#" * 80)

    first_drift_rows = []
    for traj in trajectories:
        if traj["collapse_type"] == "benign":
            continue
        deviation_step = traj.get("deviation_step", None)
        if deviation_step is None:
            continue

        # 数据集里 deviation_step 看起来是从 1 开始计数
        row = next(
            (
                r for r in all_step_results
                if r["trajectory_id"] == traj["trajectory_id"]
                and r["step_index"] == deviation_step
            ),
            None
        )
        if row is not None:
            first_drift_rows.append(row)

    first_drift_stats = {
        "n": len(first_drift_rows),
        "mean_active_count": safe_mean([r["active_feature_count"] for r in first_drift_rows]),
        "std_active_count": safe_std([r["active_feature_count"] for r in first_drift_rows]),
        "feature_activation": {},
        "by_type": {},
    }

    print(f"\n[first_drift_overall] n={len(first_drift_rows)}")
    print(f"mean_active_count: {first_drift_stats['mean_active_count']:.3f}")
    print(f"std_active_count : {first_drift_stats['std_active_count']:.3f}")

    for feat in feature_names:
        rate = safe_mean([r[feat] for r in first_drift_rows])
        first_drift_stats["feature_activation"][feat] = rate
        print(f"{feat:35s}: {rate:.3f}")

    by_type_rows = defaultdict(list)
    for r in first_drift_rows:
        by_type_rows[r["collapse_type"]].append(r)

    print("\n[first_drift_by_type]")
    for ctype, rows in by_type_rows.items():
        first_drift_stats["by_type"][ctype] = {
            "n": len(rows),
            "mean_active_count": safe_mean([r["active_feature_count"] for r in rows]),
            "feature_activation": {},
        }
        print(f"\n[{ctype}] n={len(rows)}")
        print(f"mean_active_count: {first_drift_stats['by_type'][ctype]['mean_active_count']:.3f}")
        for feat in feature_names:
            rate = safe_mean([r[feat] for r in rows])
            first_drift_stats["by_type"][ctype]["feature_activation"][feat] = rate
            print(f"{feat:35s}: {rate:.3f}")

    with open("data/atomic_first_drift_stats.json", "w", encoding="utf-8") as f:
        json.dump(first_drift_stats, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print("- data/atomic_step_results.jsonl")
    print("- data/atomic_feature_summary.json")
    print("- data/atomic_feature_binary_summary.json")
    print("- data/atomic_feature_distribution_stats.json")
    print("- data/atomic_feature_saturation_stats.json")
    print("- data/atomic_first_drift_stats.json")


if __name__ == "__main__":
    main()