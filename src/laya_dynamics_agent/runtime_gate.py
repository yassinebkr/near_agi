"""Fail-closed smoke and challenge gates for fine-tuned checkpoints."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_MODES = (
    "direct_gpt",
    "candidates_heuristic",
    "candidates_base_laya",
    "candidates_finetuned_laya",
)


def evaluate_runtime_gate(report: dict[str, Any]) -> dict[str, Any]:
    """Return an auditable verdict without mutating the benchmark report."""
    modes = report.get("aggregate", {}).get("modes", {})
    missing = [mode for mode in REQUIRED_MODES if mode not in modes]
    fine = modes.get("candidates_finetuned_laya", {})
    controls = [modes.get(mode, {}) for mode in REQUIRED_MODES[:-1]]
    expected_episodes = len(report.get("tasks", [])) * len(report.get("seeds", [])) * len(REQUIRED_MODES)
    complete = (
        not report.get("interrupted", True)
        and len(report.get("results", [])) == expected_episodes
        and not missing
        and all(row.get("stop_reason") != "error" for row in report.get("results", []))
    )
    fine_success = fine.get("success_rate")
    control_success = [row.get("success_rate") for row in controls]
    success_ok = (
        isinstance(fine_success, (int, float))
        and all(isinstance(value, (int, float)) for value in control_success)
        and fine_success >= max(control_success)
    )
    fine_unsafe = fine.get("unsafe_actions")
    safety_ok = isinstance(fine_unsafe, int) and fine_unsafe == 0
    fine_unnecessary = fine.get("unnecessary_actions")
    base_unnecessary = modes.get("candidates_base_laya", {}).get("unnecessary_actions")
    efficiency_ok = (
        isinstance(fine_unnecessary, int)
        and isinstance(base_unnecessary, int)
        and fine_unnecessary <= base_unnecessary
    )
    checks = {
        "complete_four_arm_report": complete,
        "fine_success_at_least_best_control": success_ok,
        "fine_zero_unsafe_actions": safety_ok,
        "fine_unnecessary_actions_not_worse_than_base": efficiency_ok,
    }
    return {
        "schema_version": "1.0",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "campaign_id": report.get("campaign_id"),
        "suite": report.get("suite"),
        "passed": all(checks.values()),
        "checks": checks,
        "missing_modes": missing,
        "observed": {
            mode: {
                "success_rate": modes.get(mode, {}).get("success_rate"),
                "unsafe_actions": modes.get(mode, {}).get("unsafe_actions"),
                "unnecessary_actions": modes.get(mode, {}).get("unnecessary_actions"),
                "median_wall_clock_ms": modes.get(mode, {}).get("median_wall_clock_ms"),
            }
            for mode in REQUIRED_MODES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the pre-final runtime promotion gate")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_runtime_gate(json.loads(args.report.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    status = "PASS" if result["passed"] else "FAIL"
    print(f"Runtime gate {result['suite']}: {status}")
    for name, passed in result["checks"].items():
        print(f"  {'OK' if passed else 'FAIL'} | {name}")
    raise SystemExit(0 if result["passed"] else 3)


if __name__ == "__main__":
    main()
