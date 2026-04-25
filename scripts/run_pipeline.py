"""Master automation runner for build, drift, and serve workflows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_yaml_config
from src.utils.file_handler import write_json_file
from src.utils.logger import configure_logging, get_logger

LOGGER = get_logger(__name__)


@dataclass
class StageResult:
    """Execution result for one pipeline stage."""

    stage: str
    command: List[str]
    started_at: float
    ended_at: float
    duration_seconds: float
    success: bool
    exit_code: int
    output_tail: str


class PipelineAutomationRunner:
    """Runs build/serve automation workflows using existing project scripts."""

    def __init__(self, automation_config: Dict[str, Any]) -> None:
        self.automation_config = automation_config
        self.fail_fast = bool(automation_config.get("automation", {}).get("fail_fast", True))
        self.report_path = Path(
            automation_config.get("automation", {}).get(
                "report_path", "artifacts/research/orchestration/automation_report.json"
            )
        )
        self.paths = automation_config.get("paths", {})
        self.commands = automation_config.get("commands", {})
        self.profiles = automation_config.get("profiles", {})

    def _run_command(self, stage: str, command: List[str], stream_output: bool = False) -> StageResult:
        """Runs one command and returns structured execution metadata."""
        started = time.time()
        LOGGER.info("stage=%s action=start command=%s", stage, " ".join(command))

        if stream_output:
            process = subprocess.run(command, text=True)
            output_tail = ""
            exit_code = int(process.returncode)
            success = exit_code == 0
        else:
            process = subprocess.run(command, capture_output=True, text=True)
            combined = (process.stdout or "") + "\n" + (process.stderr or "")
            output_tail = "\n".join(combined.strip().splitlines()[-20:])
            exit_code = int(process.returncode)
            success = exit_code == 0

        ended = time.time()
        result = StageResult(
            stage=stage,
            command=command,
            started_at=started,
            ended_at=ended,
            duration_seconds=ended - started,
            success=success,
            exit_code=exit_code,
            output_tail=output_tail,
        )
        LOGGER.info(
            "stage=%s action=complete success=%s exit_code=%d duration=%.2fs",
            stage,
            success,
            exit_code,
            result.duration_seconds,
        )
        return result

    def _build_command(self, profile_name: str, run_name: str) -> List[str]:
        """Builds orchestration command from config and profile settings."""
        profile = self.profiles.get(profile_name, {})
        command = [
            sys.executable,
            self.commands.get("orchestration_script", "scripts/run_orchestration.py"),
            "--config",
            self.paths.get("config", "config/config.yaml"),
            "--model-config",
            self.paths.get("model_config", "config/model_config.yaml"),
            "--run-name",
            run_name,
        ]
        if bool(profile.get("run_mlflow", False)):
            command.append("--mlflow")
        if not bool(profile.get("run_exp2", True)):
            command.append("--skip-exp2")
        if not bool(profile.get("run_deployment", True)):
            command.append("--skip-deployment")
        return command

    def _drift_command(self) -> List[str]:
        """Builds drift-check command from config."""
        return [
            sys.executable,
            self.commands.get("drift_script", "scripts/run_drift_check.py"),
            "--config",
            self.paths.get("drift_config", "config/drift_config.yaml"),
        ]

    def _serve_command(self) -> List[str]:
        """Builds API serve command from config."""
        return [
            sys.executable,
            self.commands.get("api_script", "scripts/run_api.py"),
            "--config",
            self.paths.get("inference_config", "config/inference_config.yaml"),
        ]

    def run_build(self, profile_name: str, run_name: str) -> Dict[str, Any]:
        """Runs orchestration + optional drift stages."""
        profile = self.profiles.get(profile_name, {})
        stage_results: List[StageResult] = []

        orchestration_result = self._run_command(
            stage="orchestration",
            command=self._build_command(profile_name=profile_name, run_name=run_name),
            stream_output=False,
        )
        stage_results.append(orchestration_result)
        if self.fail_fast and not orchestration_result.success:
            return self._finalize(mode="build", profile=profile_name, stage_results=stage_results)

        if bool(profile.get("run_drift", True)):
            drift_result = self._run_command(
                stage="drift_check",
                command=self._drift_command(),
                stream_output=False,
            )
            stage_results.append(drift_result)
            if self.fail_fast and not drift_result.success:
                return self._finalize(mode="build", profile=profile_name, stage_results=stage_results)

        return self._finalize(mode="build", profile=profile_name, stage_results=stage_results)

    def run_serve(self) -> Dict[str, Any]:
        """Runs API service (blocking command)."""
        stage_results: List[StageResult] = []
        serve_result = self._run_command(
            stage="serve_api",
            command=self._serve_command(),
            stream_output=True,
        )
        stage_results.append(serve_result)
        return self._finalize(mode="serve", profile="n/a", stage_results=stage_results)

    def run_full(self, profile_name: str, run_name: str) -> Dict[str, Any]:
        """Runs build workflow then starts API service."""
        report = self.run_build(profile_name=profile_name, run_name=run_name)
        if report.get("success", False):
            serve_report = self.run_serve()
            report["serve"] = serve_report
            report["success"] = bool(serve_report.get("success", False))
            write_json_file(self.report_path, report)
        return report

    def _finalize(self, mode: str, profile: str, stage_results: List[StageResult]) -> Dict[str, Any]:
        """Converts stage results to final report and writes it."""
        report = {
            "mode": mode,
            "profile": profile,
            "success": all(result.success for result in stage_results),
            "stages": [asdict(result) for result in stage_results],
        }
        write_json_file(self.report_path, report)
        return report


def parse_args() -> argparse.Namespace:
    """Parses command-line args for master automation runner."""
    parser = argparse.ArgumentParser(description="Run HX-IDL pipeline automation")
    parser.add_argument(
        "mode",
        choices=["build", "serve", "full"],
        help="build: offline pipeline, serve: API runtime, full: build then serve",
    )
    parser.add_argument("--automation-config", type=str, default="config/automation_config.yaml")
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--run-name", type=str, default="automation_run")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--plain-logs", action="store_true", help="Disable JSON logs")
    return parser.parse_args()


def main() -> None:
    """Entrypoint for automation runner."""
    args = parse_args()
    configure_logging(level=args.log_level, json_output=not args.plain_logs)

    cfg = load_yaml_config(args.automation_config)
    runner = PipelineAutomationRunner(cfg)

    if args.mode == "build":
        report = runner.run_build(profile_name=args.profile, run_name=args.run_name)
    elif args.mode == "serve":
        report = runner.run_serve()
    else:
        report = runner.run_full(profile_name=args.profile, run_name=args.run_name)

    print(json.dumps(report, indent=2))
    if not bool(report.get("success", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
