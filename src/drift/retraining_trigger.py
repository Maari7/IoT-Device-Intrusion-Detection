"""Retraining trigger logic based on drift severity and policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class RetrainingDecision:
    """Encapsulates a retraining decision."""

    trigger: bool
    severity: str
    reason: str

    def to_dict(self) -> Dict[str, str | bool]:
        """Serializes the decision."""
        return {"trigger": self.trigger, "severity": self.severity, "reason": self.reason}


class RetrainingTrigger:
    """Evaluates whether drift should trigger retraining."""

    def __init__(self, config: Dict) -> None:
        retraining_cfg = config.get("drift", {}).get("retraining", {})
        self.enabled = bool(retraining_cfg.get("enabled", True))
        self.auto_trigger = bool(retraining_cfg.get("auto_trigger", False))
        self.min_severity = str(retraining_cfg.get("min_severity", "high"))
        self.severity_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    def evaluate(self, report: Dict) -> RetrainingDecision:
        """Evaluates a drift report and returns retraining decision."""
        severity = str(report.get("severity", "low"))
        if not self.enabled:
            return RetrainingDecision(trigger=False, severity=severity, reason="retraining disabled")
        should_trigger = self.severity_rank.get(severity, 0) >= self.severity_rank.get(self.min_severity, 2)
        reason = "severity threshold reached" if should_trigger else "severity below threshold"
        return RetrainingDecision(trigger=bool(should_trigger and self.auto_trigger), severity=severity, reason=reason)

    @staticmethod
    def save(decision: RetrainingDecision, output_path: str | Path) -> Path:
        """Writes retraining decision to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(decision.to_dict(), indent=2), encoding="utf-8")
        return path
