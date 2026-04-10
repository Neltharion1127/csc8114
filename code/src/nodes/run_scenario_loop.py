#!/usr/bin/env python3
"""
Scenario loop entry-point for fsl-server and fsl-client containers.

When SCENARIO_ID is NOT set in the environment:
  Reads all scenarios + seeds from config.yaml (experiment_matrix section)
  and runs them sequentially, each with its own SCENARIO_ID subdirectory
  under bestweights/ and results/.

When SCENARIO_ID IS set:
  Falls through directly to server_node / client_node (single-scenario mode).

Usage (docker-compose command):
  python src/nodes/run_scenario_loop.py --role server
  python src/nodes/run_scenario_loop.py --role client
"""
import argparse
import copy
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """In-place deep merge of override into base."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def _run_node(module: str, env: dict[str, str]) -> int:
    """Run a Python module as a subprocess and return its exit code."""
    result = subprocess.run(
        [sys.executable, "-u", "-m", module],
        cwd=PROJECT_ROOT,
        env=env,
    )
    return result.returncode


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="FSL Scenario Loop Runner")
    parser.add_argument(
        "--role",
        choices=["server", "client"],
        required=True,
        help="Which node role this container plays.",
    )
    args = parser.parse_args()

    scenario_id = os.environ.get("SCENARIO_ID", "").strip()
    node_module = (
        "src.nodes.server_node" if args.role == "server" else "src.nodes.client_node"
    )

    # ── Single-scenario mode: SCENARIO_ID already provided ────────────────────
    if scenario_id:
        print(f"[SCENARIO LOOP] Single-scenario mode: role={args.role} SCENARIO_ID={scenario_id}")
        # Replace this process with the node process directly (no subprocess overhead).
        os.execv(sys.executable, [sys.executable, "-u", "-m", node_module])
        return  # unreachable; execv replaces this process

    # ── Multi-scenario mode: read scenarios from config.yaml ──────────────────
    config_path = Path(
        os.environ.get("FSL_CONFIG_PATH", str(PROJECT_ROOT / "config.yaml"))
    )
    with config_path.open("r", encoding="utf-8") as fh:
        root_cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    matrix_cfg = root_cfg.get("experiment_matrix", {})
    scenarios: list[dict[str, Any]] = matrix_cfg.get("scenarios", [])

    if not scenarios:
        print("[SCENARIO LOOP] No scenarios defined in config — running single session.")
        os.execv(sys.executable, [sys.executable, "-u", "-m", node_module])
        return

    raw_seeds = matrix_cfg.get("seeds", [root_cfg.get("training", {}).get("seed", 42)])
    seeds = [int(s) for s in raw_seeds]

    # Use a single shared session ID for this entire matrix run.
    session_id = (
        os.environ.get("SESSION_ID", "").strip()
        or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    print(
        f"[SCENARIO LOOP] role={args.role} | session={session_id} "
        f"| scenarios={len(scenarios)} | seeds={seeds}"
    )

    # Write per-scenario merged configs to a temp directory.
    tmp_dir = PROJECT_ROOT / "results" / "matrix_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []

    for seed in seeds:
        for scenario in scenarios:
            sid = str(scenario.get("id", "unknown")).strip()
            desc = str(scenario.get("description", ""))
            run_id = f"{sid}_seed{seed}"
            print(f"\n[SCENARIO LOOP] ── {run_id}: {desc}")

            # Build merged config: base + scenario overrides + seed.
            run_cfg = copy.deepcopy(root_cfg)
            _deep_merge(run_cfg, scenario.get("overrides", {}))
            run_cfg.setdefault("training", {})["seed"] = seed

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tmp_config = tmp_dir / f"{stamp}_{run_id}.yaml"
            with tmp_config.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(run_cfg, fh, sort_keys=False)

            env = os.environ.copy()
            env["SESSION_ID"] = session_id
            env["SCENARIO_ID"] = run_id  # Use run_id (e.g. '01_seed42') to separate seed paths
            env["FSL_CONFIG_PATH"] = str(tmp_config)

            rc = _run_node(node_module, env)
            if rc != 0:
                print(
                    f"[SCENARIO LOOP] {run_id} exited with code {rc}. "
                    "Continuing to next scenario..."
                )
                failed.append(run_id)
            else:
                print(f"[SCENARIO LOOP] {run_id} completed successfully.")

    total = len(scenarios) * len(seeds)
    ok = total - len(failed)
    print(f"\n[SCENARIO LOOP] Done: {ok}/{total} OK | session={session_id}")
    if failed:
        print(f"[SCENARIO LOOP] Failed runs: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
