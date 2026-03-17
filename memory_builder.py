from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Set


@dataclass
class IncidentMemoryState:
    core_incident_objects: List[str] = field(default_factory=list)
    anchored_objects: List[str] = field(default_factory=list)
    observed_findings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IncidentMemoryBuilder:
    """
    Minimal memory builder.

    Uses only:
      - task_text
      - steps[:current_step_index]

    Does NOT read future steps.
    """

    OBJECT_PATTERNS = [
        r"\bbattery pack [a-z]\d+\b",
        r"\bpack [a-z]\d+\b",
        r"\bcell [a-z]\d+(?:-\d+)?\b",
        r"\binv-\d+\b",
        r"\binverter [a-z]+-\d+\b",
        r"\bcb-\d+\b",
        r"\bcombiner box\b",
        r"\bpv array\b",
        r"\bcooling fan\b",
        r"\bair filter\b",
        r"\bjunction terminals?\b",
        r"\bcommunication panel\b",
        r"\bethernet cable\b",
        r"\bethernet switch\b",
        r"\bdc bus\b",
        r"\bcontroller\b",
        r"\bcontrol board\b",
        r"\bfirmware\b",
        r"\bgrid operator\b",
        r"\bgrid connection point\b",
        r"\bess\b",
        r"\bscada\b",
        r"\bvlan \d+\b",
        r"\bport [a-z]+\d+/\d+/\d+\b",
        r"\bmodbus\b",
    ]

    def build(self, trajectory: Dict[str, Any], current_step_index: int) -> IncidentMemoryState:
        task_text = trajectory.get("task_text", "")
        steps = trajectory.get("steps", [])
        history_steps = steps[:current_step_index]

        core_incident_objects = self._extract_objects(task_text)

        anchored_objects: List[str] = []
        anchored_seen: Set[str] = set()

        observed_findings: List[str] = []
        finding_seen: Set[str] = set()

        for step in history_steps:
            observation_text = step.get("observation_text", "")
            evidence_text = step.get("evidence_text", "")

            # anchored_objects only from historical observation/evidence
            hist_text = f"{observation_text} {evidence_text}"
            for obj in self._extract_objects(hist_text):
                if obj not in anchored_seen:
                    anchored_seen.add(obj)
                    anchored_objects.append(obj)

            for finding in self._extract_findings(observation_text, evidence_text):
                if finding not in finding_seen:
                    finding_seen.add(finding)
                    observed_findings.append(finding)

        return IncidentMemoryState(
            core_incident_objects=core_incident_objects,
            anchored_objects=anchored_objects,
            observed_findings=observed_findings,
        )

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _extract_objects(self, text: str) -> List[str]:
        text = self._normalize(text)
        found: List[str] = []
        seen: Set[str] = set()

        for pattern in self.OBJECT_PATTERNS:
            for match in re.finditer(pattern, text):
                obj = match.group(0).strip()
                if obj not in seen:
                    seen.add(obj)
                    found.append(obj)

        return found

    def _extract_findings(self, observation_text: str, evidence_text: str) -> List[str]:
        """
        Keep this very lightweight and deterministic.
        We store short normalized findings from historical observation/evidence.
        """
        text = self._normalize(f"{observation_text}. {evidence_text}")
        sentences = [s.strip() for s in re.split(r"[.;]", text) if s.strip()]

        findings: List[str] = []
        for s in sentences:
            # only keep moderately informative sentences
            if len(s) < 12:
                continue
            findings.append(s[:160])

        return findings[:4]  # keep small, lightweight