from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class IncidentMemoryState:
    core_incident_objects: List[str] = field(default_factory=list)
    incident_anchored_objects: List[str] = field(default_factory=list)
    historically_seen_objects: List[str] = field(default_factory=list)
    observed_findings: List[str] = field(default_factory=list)
    raw_core_response: str = ""
    raw_history_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IncidentMemoryBuilder:
    """
    Two-stage pure-LLM memory builder.

    Stage 1:
      task_text -> core_incident_objects

    Stage 2:
      historical observation/evidence ->
        - incident_anchored_objects
        - historically_seen_objects
        - observed_findings

    Only uses:
      - task_text
      - steps[:t]

    Does NOT read future steps.
    Does NOT treat action text as evidence.
    """

    def __init__(self, llm_call: Callable[[List[Dict[str, str]]], str]) -> None:
        self.llm_call = llm_call

    def build(self, trajectory: Dict[str, Any], current_step_index: int) -> IncidentMemoryState:
        task_text = trajectory.get("task_text", "")
        steps = trajectory.get("steps", [])
        history_steps = steps[:current_step_index]

        core_raw = self._extract_core_incident_objects(task_text)
        core_parsed = self._parse_json(core_raw) or {}
        core_incident_objects = self._clean_list(core_parsed.get("core_incident_objects", []))

        history_raw = self._extract_history_memory(task_text, history_steps)
        history_parsed = self._parse_json(history_raw) or {}

        incident_anchored_objects = self._clean_list(
            history_parsed.get("incident_anchored_objects", [])
        )
        historically_seen_objects = self._clean_list(
            history_parsed.get("historically_seen_objects", [])
        )
        observed_findings = self._clean_list(
            history_parsed.get("observed_findings", [])
        )

        return IncidentMemoryState(
            core_incident_objects=core_incident_objects,
            incident_anchored_objects=incident_anchored_objects,
            historically_seen_objects=historically_seen_objects,
            observed_findings=observed_findings,
            raw_core_response=core_raw or "",
            raw_history_response=history_raw or "",
        )

    def _extract_core_incident_objects(self, task_text: str) -> str:
        system_prompt = (
            "You are a strict JSON-only extractor.\n"
            "Output only valid JSON. No markdown. No explanation.\n"
            "Start with { and end with }.\n"
        )

        user_prompt = f"""
TASK:
{task_text}

Extract only the primary incident object(s) explicitly defined by the task.
Keep the list very small and stable.
Prefer the main device/component/problem target.

Return exactly:
{{
  "core_incident_objects": ["..."]
}}
""".strip()

        raw = self.llm_call([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        print("DEBUG core_raw:", repr((raw or "")[:300]))
        return raw or ""

    def _extract_history_memory(self, task_text: str, history_steps: List[Dict[str, Any]]) -> str:
        history_text = self._format_history_observation_evidence(history_steps)

        system_prompt = (
            "You are a strict JSON-only incident memory summarizer.\n"
            "Use only historical observation_text and evidence_text.\n"
            "Do not use action_text.\n"
            "Do not use goal_text.\n"
            "Do not infer hidden facts.\n"
            "Do not use future information.\n"
            "Output only valid JSON. No markdown. No explanation.\n"
        )

        user_prompt = f"""
TASK:
{task_text}

HISTORICAL OBSERVATION/EVIDENCE:
{history_text}

Build exactly these fields:

1. incident_anchored_objects:
   Objects that appeared in historical observation/evidence AND are still directly tied to the original incident.
   Examples: the faulty device, directly relevant components, directly relevant measurements.
   Do NOT include people, reports, schedules, procurement objects, or broader audit/security artifacts unless they are clearly part of the core incident itself.

2. historically_seen_objects:
   Objects that appeared in historical observation/evidence, even if they are not part of the core incident.
   This can include people, files, reports, records, etc.

3. observed_findings:
   Short factual findings already established by historical observation/evidence.
   Keep only incident-relevant findings.
   Keep them short and concrete.
   Do not include speculative diagnoses unless explicitly stated in observation/evidence.
   Prefer at most 5 findings.

If there is no history, return empty lists.

Return exactly:
{{
  "incident_anchored_objects": ["..."],
  "historically_seen_objects": ["..."],
  "observed_findings": ["..."]
}}
""".strip()

        raw = self.llm_call([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        print("DEBUG history_raw:", repr((raw or "")[:300]))
        return raw or ""

    def _format_history_observation_evidence(self, history_steps: List[Dict[str, Any]]) -> str:
        if not history_steps:
            return "(none)"

        chunks: List[str] = []
        for step in history_steps:
            idx = step.get("step_index", "?")
            observation_text = step.get("observation_text", "")
            evidence_text = step.get("evidence_text", "")

            chunks.append(
                f"Step {idx}:\n"
                f"Observation: {observation_text}\n"
                f"Evidence: {evidence_text}"
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