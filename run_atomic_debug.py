#!/usr/bin/env python3
"""Debug script for memory_builder output - Step A, C, D."""

import json
from openai import OpenAI

import memory_builder
import atomic_features

print("=" * 60)
print("STEP A: 确认导入路径")
print("=" * 60)
print(f"memory_builder.__file__ = {memory_builder.__file__}")
print(f"atomic_features.__file__ = {atomic_features.__file__}")
print()

DATA_PATH = "data/ontosync_v1_testset.jsonl"
MODEL = "/home/skyhe666/llm-model/Qwen/Qwen3.5-9B"

client = OpenAI(api_key="EMPTY", base_url="http://10.1.47.245:8000/v1")


def llm_call(messages):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=3000,
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


def main():
    trajectories = load_jsonl(DATA_PATH)

    memory_builder_instance = memory_builder.IncidentMemoryBuilder(llm_call=llm_call)
    extractor = atomic_features.AtomicFeatureExtractor(llm_call=llm_call)

    print("=" * 60)
    print("STEP D: 只跑第一条轨迹的前两步")
    print("=" * 60)

    # 只取第一条轨迹
    traj = trajectories[0]
    traj_id = traj["trajectory_id"]
    collapse_type = traj["collapse_type"]
    label = traj["label"]

    print(f"Trajectory: {traj_id} | type={collapse_type} | label={label}")
    print(f"Task text: {traj.get('task_text', '')[:200]}...")
    print()

    # 只跑 step 1 和 step 2 (step_idx = 0 和 1)
    for step_idx in [0, 1]:
        print("=" * 60)
        print(f"STEP {step_idx + 1}")
        print("=" * 60)

        step = traj["steps"][step_idx]

        # 构建 memory
        memory_obj = memory_builder_instance.build(traj, step_idx)
        memory_state = memory_obj.to_dict()

        # 步骤C: 打印 memory 原始返回状态
        print("\n--- STEP C: Memory State Debug ---")
        print(f"Memory core: {memory_state.get('core_incident_objects', [])}")
        print(f"Memory anchored: {memory_state.get('anchored_objects', [])}")
        print(f"Memory findings: {memory_state.get('observed_findings', [])}")
        print(f"bool(raw_core_response): {bool(memory_state.get('raw_core_response'))}")
        print(f"bool(raw_history_response): {bool(memory_state.get('raw_history_response'))}")

        if memory_state.get("raw_core_response"):
            print(f"\n--- RAW CORE (first 400 chars) ---")
            print(repr(memory_state["raw_core_response"][:400]))

        if memory_state.get("raw_history_response"):
            print(f"\n--- RAW HISTORY (first 400 chars) ---")
            print(repr(memory_state["raw_history_response"][:400]))

        # 提取 atomic features
        result = extractor.extract_step(step, traj, memory_state)
        result_dict = result.to_dict()

        print(f"\n--- Atomic Features ---")
        print(json.dumps({k: v for k, v in result_dict.items() if k != 'raw_response'}, ensure_ascii=False))

        print(f"\n--- Step Info ---")
        print(f"Action: {step['action_text'][:100]}...")
        print(f"Goal: {step.get('goal_text', '')[:100]}...")
        print(f"Evidence: {step.get('evidence_text', '')[:100]}...")

    print("\n" + "=" * 60)
    print("调试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
