#!/usr/bin/env python3
"""
Scenario loop entry-point for fsl-server and fsl-client nodes.
Automatically runs multiple scenarios from matrix.yaml.
"""
import argparse
import copy
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """In-place deep merge of override into base."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)

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

    # ── Single-scenario mode: SCENARIO_ID already provided ──
    if scenario_id:
        print(f"[SCENARIO LOOP] Single-scenario mode: role={args.role} SCENARIO_ID={scenario_id}")
        os.execv(sys.executable, [sys.executable, "-u", "-m", node_module])
        return

    # ── Multi-scenario mode: read matrix ──
    config_path = Path(os.environ.get("FSL_CONFIG_PATH", str(PROJECT_ROOT / "config.yaml")))
    matrix_path = Path(os.environ.get("FSL_MATRIX_CONFIG_PATH", str(PROJECT_ROOT / "matrix.yaml")))
    
    with config_path.open("r", encoding="utf-8") as fh:
        root_cfg = yaml.safe_load(fh) or {}
    with matrix_path.open("r", encoding="utf-8") as fh:
        matrix_raw = yaml.safe_load(fh) or {}

    matrix_cfg = matrix_raw.get("experiment_matrix", {})
    scenarios = matrix_cfg.get("scenarios", [])
    raw_seeds = matrix_cfg.get("seeds", [root_cfg.get("training", {}).get("seed", 42)])
    seeds = [int(s) for s in raw_seeds]
    session_id = os.environ.get("SESSION_ID", "").strip() or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    tmp_dir = PROJECT_ROOT / "results" / "matrix_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    run_count = 0
    total_runs = len(scenarios) * len(seeds)

    for seed in seeds:
        for scenario in scenarios:
            if run_count > 0:
                print("\n" + "─"*60)
                print(f"[SCENARIO LOOP] Waiting 5 seconds for system cleanup...")
                time.sleep(5)

            sid = str(scenario.get("id", "unknown")).strip()
            run_id = f"{sid}_seed{seed}"
            run_count += 1
            print(f"\n[SCENARIO LOOP] ── {run_count}/{total_runs}: Starting {run_id} ──")

            # Prepare merged config
            run_cfg = copy.deepcopy(root_cfg)
            _deep_merge(run_cfg, scenario.get("overrides", {}))
            run_cfg.setdefault("training", {})["seed"] = seed
            tmp_config = tmp_dir / f"{run_id}_cfg.yaml"
            with tmp_config.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(run_cfg, fh, sort_keys=False)


            env = os.environ.copy()
            env["SESSION_ID"] = session_id
            env["SCENARIO_ID"] = run_id
            env["FSL_CONFIG_PATH"] = str(tmp_config)

            # --- Synchronisation Barrier (Client Only) ---
            if args.role == "client":
                # Prioritize environment variable over config file for distributed runs
                server_host = os.environ.get("FSL_SERVER_HOST") or run_cfg.get("grpc", {}).get("server_host", "localhost")
                server_port = run_cfg.get("grpc", {}).get("server_port", 50051)
                
                print(f"[SCENARIO LOOP] Client waiting for Server ({server_host}:{server_port}) to switch to {run_id}...")
                
                # Use dynamic loading to find files exactly where Server/Client nodes expect them
                import importlib.util
                def load_module(name, path):
                    spec = importlib.util.spec_from_file_location(name, path)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    return mod

                try:
                    fsl_pb2 = load_module("fsl_pb2", str(PROJECT_ROOT / "proto" / "fsl_pb2.py"))
                    fsl_pb2_grpc = load_module("fsl_pb2_grpc", str(PROJECT_ROOT / "proto" / "fsl_pb2_grpc.py"))
                    import grpc
                except Exception as e:
                    print(f"[SCENARIO LOOP] Critical: Could not find gRPC generated files in proto/: {e}")
                    sys.exit(1)
                
                while True:
                    try:
                        channel = grpc.insecure_channel(f"{server_host}:{server_port}")
                        stub = fsl_pb2_grpc.FSLServiceStub(channel)
                        # Use the Heartbeat/GetInfo endpoint to ask the server its current ID
                        empty_msg = getattr(fsl_pb2, 'Empty')()
                        response = stub.GetInfo(empty_msg, timeout=2.0)
                        current_server_sid = getattr(response, 'scenario_id', 'unknown')
                        
                        if current_server_sid == run_id:
                            print(f"[SCENARIO LOOP] Server is READY for {run_id}. Proceeding.")
                            break
                        else:
                            print(f"[SCENARIO LOOP] Server is in mismatching scenario: {current_server_sid} != {run_id}. Waiting 3s...")
                    except Exception:
                        # Server might be offline or starting up
                        print(f"[SCENARIO LOOP] Server ({server_host}:{server_port}) is offline or restarting. Waiting 3s...")
                    
                    time.sleep(3)

            # --- Start Subprocess with Logging ---
            log_dir = PROJECT_ROOT / "results" / "logs" / session_id
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = log_dir / f"{run_id}.log"
            
            print(f"[SCENARIO LOOP] Logging to: {log_file_path}")
            
            try:
                # We use pipe to capture output so we can save it to file AND show it on console
                proc = subprocess.Popen(
                    [sys.executable, "-u", "-m", node_module],
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                # Setup signal handling
                def handle_signal(sig, frame):
                    print(f"\n[SCENARIO LOOP] Signal received. Terminating {run_id}...")
                    proc.terminate()
                    sys.exit(0)
                
                signal.signal(signal.SIGINT, handle_signal)
                signal.signal(signal.SIGTERM, handle_signal)
                
                # Use a file to save the log
                with open(log_file_path, "w", encoding="utf-8") as log_fh:
                    # Stream output line by line
                    for line in proc.stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        log_fh.write(line)
                        log_fh.flush()
                
                rc = proc.wait()
            except Exception as e:
                print(f"[SCENARIO LOOP] Error during subprocess execution: {e}")
                rc = -1

            if rc != 0:
                print(f"[SCENARIO LOOP] {run_id} failed (code {rc}). Continuing...")
                failed.append(run_id)
            else:
                print(f"[SCENARIO LOOP] {run_id} completed successfully.")

    print(f"\n[SCENARIO LOOP] All done: {total_runs - len(failed)}/{total_runs} OK")
    if failed:
        sys.exit(1)

if __name__ == "__main__":
    main()
