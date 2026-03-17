from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class IncidentMemoryState:
    core_incident_objects: List[str] = field(default_factory=list)
    anchored_objects: List[str] = field(default_factory=list)
    observed_findings: List[str] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IncidentMemoryBuilder:
    """
    Pure-LLM memory builder.

    For step t, it uses only:
      - task_text
      - steps[:t]

    It does NOT read future steps.
    It does NOT treat action text as evidence.
    """

    def __init__(self, llm_call: Callable[[List[Dict[str, str]]], str]) -> None:
        self.llm_call = llm_call

    def build(self, trajectory: Dict[str, Any], current_step_index: int) -> IncidentMemoryState:
        task_text = trajectory.get("task_text", "")
        steps = trajectory.get("steps", [])
        history_steps = steps[:current_step_index]

        history_text = self._format_history(history_steps)

        system_prompt = (
            "You are a strict incident-memory builder.\n"
            "Your job is to summarize the already-established incident context BEFORE the current step.\n"
            "You must only use the task text and historical steps provided.\n"
            "Do NOT read future information.\n"
            "Do NOT infer hidden facts.\n"
            "Do NOT treat actions, goals, or hypotheses as evidence.\n"
            "Only observation_text and evidence_text can anchor objects or findings.\n"
            "Return strict JSON only.\n"
        )

        user_prompt = f"""
TASK:
{task_text}

HISTORICAL STEPS BEFORE CURRENT STEP:
{history_text}

Build a lightweight memory state with exactly these fields:

1. core_incident_objects:
   The main incident objects explicitly defined by the task.
   Keep this list small and stable.
   Do NOT include every mentioned object.
   Prefer the primary device/component/problem target from the task.

2. anchored_objects:
   Objects that have already appeared in historical observation_text or evidence_text.
   Do NOT include objects that appear only in action_text or goal_text.
   Do NOT include speculative or hypothetical targets.

3. observed_findings:
   Short factual findings already established by historical observation_text or evidence_text.
   Each finding should be short, concrete, and evidence-grounded.
   Do NOT include future outcomes.
   Do NOT include inferred diagnoses unless explicitly observed/evidenced.

Important rules:
- If history is empty, anchored_objects and observed_findings should be empty lists.
- Keep each list concise.
- Remove duplicates.
- Normalize wording where reasonable.
- Return strict JSON only.

Return exactly this schema:
{{
  "core_incident_objects": ["..."],
  "anchored_objects": ["..."],
  "observed_findings": ["..."]
}}
""".strip()

        raw = self.llm_call([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        parsed = self._parse_json(raw)
        if parsed is None:
            return IncidentMemoryState(raw_response=raw)

        return IncidentMemoryState(
            core_incident_objects=self._clean_list(parsed.get("core_incident_objects", [])),
            anchored_objects=self._clean_list(parsed.get("anchored_objects", [])),
            observed_findings=self._clean_list(parsed.get("observed_findings", [])),
            raw_response=raw,
        )

    def _format_history(self, history_steps: List[Dict[str, Any]]) -> str:
        if not history_steps:
            return "(none)"

        chunks: List[str] = []
        for step in history_steps:
            idx = step.get("step_index", "?")
            observation_text = step.get("observation_text", "")
            evidence_text = step.get("evidence_text", "")

            chunks.append(
                f"Step {idx}:\n"
                f"  Observation: {observation_text}\n"
                f"  Evidence: {evidence_text}"
            )
        return "\n\n".join(chunks)

    def _parse_json(self, raw: str) -> Optional[Dict[str, Any]]:
        if not raw:
            return None

        text = raw.strip()

        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        try:
            return json.loads(text)
        except Exception:
            return None

    def _clean_list(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []

        out: List[str] = []
        seen = set()

        for item in value:
            if not isinstance(item, str):
                continue
            s = " ".join(item.strip().split())
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)

        return out