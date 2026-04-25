"""Report builders for hybrid experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def _format_metrics(title: str, metrics: Dict[str, float]) -> str:
    lines = [f"### {title}"]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value:.6f}")
    return "\n".join(lines)


def build_markdown_report(
    output_path: str | Path,
    experiment_name: str,
    baseline_metrics: Dict[str, float],
    hybrid_metrics: Dict[str, float],
    fusion_metrics: Dict[str, float],
    artifacts: Dict[str, str],
) -> Path:
    """Writes markdown report with metrics and generated artifacts."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sections = [
        f"# Experiment Report: {experiment_name}",
        "",
        _format_metrics("Baseline Multiclass (XGBoost)", baseline_metrics),
        "",
        _format_metrics("Hybrid Binary (Dual Consistency)", hybrid_metrics),
        "",
        _format_metrics("Fusion Binary", fusion_metrics),
        "",
        "### Artifacts",
    ]

    for key, value in artifacts.items():
        sections.append(f"- {key}: {value}")

    path.write_text("\n".join(sections), encoding="utf-8")
    return path
