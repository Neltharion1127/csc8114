import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _to_container_config_path(host_path: Path) -> str:
    """Map a host config path under project root to the container path (/app/...)."""
    rel = host_path.resolve().relative_to(PROJECT_ROOT.resolve())
    return f"/app/{rel.as_posix()}"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a dict: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _list_sessions(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("20")}


def _detect_session(
    before_bw: set[str],
    after_bw: set[str],
    before_res: set[str],
    after_res: set[str],
) -> str:
    new_bw = after_bw - before_bw
    new_res = after_res - before_res
    new_common = sorted(new_bw & new_res)
    if new_common:
        return new_common[-1]
    if new_bw:
        return sorted(new_bw)[-1]
    if new_res:
        return sorted(new_res)[-1]
    common_after = sorted(after_bw & after_res)
    if common_after:
        return common_after[-1]
    raise RuntimeError("Cannot detect new session from bestweights/results.")


def _find_eval_json(results_dir: Path, session_id: str) -> Path:
    direct = results_dir / session_id / f"evaluation_report_{session_id}.json"
    if direct.exists():
        return direct
    candidates = sorted((results_dir / session_id).glob("evaluation_report_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No evaluation_report_*.json in results/{session_id}")
    return candidates[-1]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_eval_metrics(eval_json_path: Path) -> dict[str, Any]:
    report = json.loads(eval_json_path.read_text(encoding="utf-8"))
    clients = report.get("clients", [])
    if not clients:
        return {}

    tp = sum(int(c.get("tp", 0)) for c in clients)
    fp = sum(int(c.get("fp", 0)) for c in clients)
    fn = sum(int(c.get("fn", 0)) for c in clients)
    tn = sum(int(c.get("tn", 0)) for c in clients)
    macro_precision = _mean([_safe_float(c.get("precision", 0.0)) for c in clients])
    macro_recall = _mean([_safe_float(c.get("recall", 0.0)) for c in clients])
    macro_f1 = _mean([_safe_float(c.get("f1", 0.0)) for c in clients])
    mse_mean = _mean([_safe_float(c.get("mse", 0.0)) for c in clients])
    mae_mean = _mean([_safe_float(c.get("mae", 0.0)) for c in clients])

    micro_precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    micro_recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    micro_f1 = (
        (2.0 * micro_precision * micro_recall / (micro_precision + micro_recall))
        if (micro_precision + micro_recall) > 0
        else 0.0
    )
    accuracy = ((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "pairing_mode": report.get("pairing_mode", ""),
        "server_round": report.get("server_round", ""),
        "clients_evaluated": len(clients),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "mse_mean": mse_mean,
        "mae_mean": mae_mean,
    }


def _read_server_metrics(session_dir: Path) -> dict[str, Any]:
    logs = sorted(session_dir.glob("server_log_*.csv"))
    if not logs:
        return {}
    with logs[-1].open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    latencies = []
    payloads = []
    compression_modes = set()
    rho_values = set()
    for row in rows:
        if row.get("reported_latency_ms", "") != "":
            latencies.append(_safe_float(row["reported_latency_ms"]))
        if row.get("payload_bytes", "") != "":
            payloads.append(_safe_float(row["payload_bytes"]))
        if row.get("compression_mode"):
            compression_modes.add(str(row["compression_mode"]))
        if row.get("next_rho"):
            rho_values.add(str(row["next_rho"]))

    return {
        "server_rows": len(rows),
        "avg_reported_latency_ms": _mean(latencies) if latencies else 0.0,
        "avg_payload_bytes": _mean(payloads) if payloads else 0.0,
        "compression_modes_seen": "|".join(sorted(compression_modes)),
        "rho_values_seen": "|".join(sorted(rho_values, key=lambda v: int(v) if v.isdigit() else v)),
    }


def _run_command(cmd: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(f"[CMD] {printable}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def _write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fieldnames = [
        "run_id",
        "scenario_id",
        "scenario_desc",
        "seed",
        "session_id",
        "status",
        "error",
        "pairing_mode",
        "server_round",
        "clients_evaluated",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "accuracy",
        "tp",
        "fp",
        "fn",
        "tn",
        "mse_mean",
        "mae_mean",
        "avg_reported_latency_ms",
        "avg_payload_bytes",
        "compression_modes_seen",
        "rho_values_seen",
        "config_path",
        "started_at",
        "ended_at",
        "duration_sec",
    ]
    extra_fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in base_fieldnames and key not in extra_fieldnames:
                extra_fieldnames.append(key)

    fieldnames = base_fieldnames + extra_fieldnames
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_existing_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _merge_rows_by_run_id(
    existing_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = list(existing_rows)
    index_by_run_id: dict[str, int] = {}
    for idx, row in enumerate(merged):
        run_id = str(row.get("run_id", "")).strip()
        if run_id:
            index_by_run_id[run_id] = idx

    for row in new_rows:
        run_id = str(row.get("run_id", "")).strip()
        if run_id and run_id in index_by_run_id:
            merged[index_by_run_id[run_id]] = row
        else:
            if run_id:
                index_by_run_id[run_id] = len(merged)
            merged.append(row)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Run experiment matrix defined in config.yaml.")
    parser.add_argument(
        "--config",
        default=os.environ.get("FSL_CONFIG_PATH", "config.yaml"),
        help="Matrix config file path (default: FSL_CONFIG_PATH or config.yaml)",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated scenario IDs to run (e.g. M01,M02)",
    )
    parser.add_argument(
        "--backend",
        choices=["native", "docker", "dist"],
        default="",
        help="Execution backend override (native, docker, or dist for VPS+Pi cluster).",
    )
    parser.add_argument("--max-runs", type=int, default=0, help="Limit total runs (0 means no limit).")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands only.")
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    root_cfg = _load_yaml(config_path)
    matrix_cfg = root_cfg.get("experiment_matrix", {})
    scenarios = matrix_cfg.get("scenarios", [])
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("config.experiment_matrix.scenarios must be a non-empty list.")

    seeds = matrix_cfg.get("seeds", [root_cfg.get("training", {}).get("seed", 42)])
    if not isinstance(seeds, list) or not seeds:
        seeds = [42]
    seeds = [int(s) for s in seeds]

    only_ids = {s.strip() for s in args.only.split(",") if s.strip()}
    if only_ids:
        scenarios = [s for s in scenarios if str(s.get("id", "")).strip() in only_ids]
    if not scenarios:
        raise ValueError("No scenarios selected.")

    runner_cfg = matrix_cfg.get("runner", {})
    server_device = str(runner_cfg.get("server_device", "cpu"))
    client_device = str(runner_cfg.get("client_device", "cpu"))
    eval_device = str(runner_cfg.get("eval_device", "cpu"))
    startup_timeout = int(runner_cfg.get("startup_timeout", 90))
    backend = str(args.backend or runner_cfg.get("backend", "native")).strip().lower()
    if backend not in {"native", "docker", "dist"}:
        raise ValueError(f"Unsupported backend: {backend}")
    summary_csv_rel = str(runner_cfg.get("summary_csv", "results/matrix_summary.csv"))
    summary_csv = _resolve_path(summary_csv_rel)

    # 🆕 Fix: Generate a unified session ID for this matrix run
    main_session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"[MATRIX] Using Unified Session ID: {main_session_id}")

    configs_dir = PROJECT_ROOT / "results" / "matrix_configs"

    plan: list[tuple[dict[str, Any], int]] = []
    for scenario in scenarios:
        for seed in seeds:
            plan.append((scenario, seed))

    if args.max_runs > 0:
        plan = plan[: args.max_runs]

    print(
        f"[MATRIX] config={config_path} | scenarios={len(scenarios)} | "
        f"seeds={len(seeds)} | total_runs={len(plan)}"
    )
    print(f"[MATRIX] backend={backend}")
    print(f"[MATRIX] summary_csv={summary_csv}")

    if args.dry_run:
        print("[MATRIX] dry-run plan:")
        for idx, (scenario, seed) in enumerate(plan, start=1):
            scenario_id = str(scenario.get("id", f"S{idx:02d}"))
            scenario_desc = str(scenario.get("description", ""))
            print(f"  - {idx:02d}. {scenario_id}_seed{seed} | {scenario_desc}")
        return 0

    configs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    existing_rows = _load_existing_summary(summary_csv)
    for idx, (scenario, seed) in enumerate(plan, start=1):
        scenario_id = str(scenario.get("id", f"S{idx:02d}"))
        scenario_desc = str(scenario.get("description", ""))
        run_id = f"{scenario_id}_seed{seed}"
        print(f"\n=== [{idx}/{len(plan)}] {run_id} ===")

        run_cfg = copy.deepcopy(root_cfg)
        overrides = scenario.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"Scenario {scenario_id} overrides must be a dict.")
        _deep_merge(run_cfg, overrides)
        run_cfg.setdefault("training", {})
        run_cfg["training"]["seed"] = int(seed)
        run_num_clients = max(
            1,
            int(
                run_cfg.get("federated", {}).get(
                    "num_clients",
                    root_cfg.get("federated", {}).get("num_clients", 3),
                )
            ),
        )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_config_path = configs_dir / f"{stamp}_{run_id}.yaml"
        with run_config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(run_cfg, f, sort_keys=False)

        env = os.environ.copy()
        env_local = env.copy()
        env_local["FSL_CONFIG_PATH"] = str(run_config_path)
        env_local["SESSION_ID"] = main_session_id # 🆕 Force same session
        env_local["SCENARIO_ID"] = scenario_id      # 🆕 Tell server which scenario
        
        env_backend = env_local.copy()
        if backend == "docker":
            env_backend["FSL_CONFIG_PATH"] = _to_container_config_path(run_config_path)
        else:
            env_backend["FSL_CONFIG_PATH"] = str(run_config_path)

        started_at = datetime.now().isoformat(timespec="seconds")
        start_ts = time.time()
        session_id = ""
        status = "ok"
        error = ""
        eval_metrics: dict[str, Any] = {}
        server_metrics: dict[str, Any] = {}

        before_bw = _list_sessions(PROJECT_ROOT / "bestweights")
        before_res = _list_sessions(PROJECT_ROOT / "results")

        # Cleanup is best-effort; do not stop batch if no process exists.
        if backend == "native":
            cleanup_cmd = ["make", "native-clean"]
        elif backend == "dist":
            cleanup_cmd = []  # dist-start handles its own teardown
        else:
            cleanup_cmd = ["make", "docker-clean"]
        if cleanup_cmd:
            subprocess.run(cleanup_cmd, cwd=PROJECT_ROOT, env=env_backend, check=False)
        try:
            if backend == "native":
                run_cmd = [
                    "make",
                    "native-run",
                    f"SERVER_DEVICE={server_device}",
                    f"CLIENT_DEVICE={client_device}",
                    f"STARTUP_TIMEOUT={startup_timeout}",
                    f"NUM_CLIENTS={run_num_clients}",
                    "AUTO_PLOT=0",
                ]
            elif backend == "dist":
                # Distributed: orchestrate VPS server + Raspberry Pi clients
                run_cmd = [
                    "make",
                    "dist-start",
                    f"SESSION_ID={main_session_id}",
                    f"SCENARIO_ID={scenario_id}",
                    f"NUM_CLIENTS={run_num_clients}",
                ]
            else:
                run_cmd = [
                    "make",
                    "docker-run",
                    f"NUM_CLIENTS={run_num_clients}",
                    "AUTO_PLOT=0",
                ]
            _run_command(
                run_cmd,
                env=env_backend,
                dry_run=args.dry_run,
            )

            if not args.dry_run:
                after_bw = _list_sessions(PROJECT_ROOT / "bestweights")
                after_res = _list_sessions(PROJECT_ROOT / "results")
                session_id = _detect_session(before_bw, after_bw, before_res, after_res)
                print(f"[MATRIX] detected session: {session_id}")

                _run_command(
                    [
                        sys.executable,
                        "-m",
                        "src.data.run_evaluation",
                        "--device",
                        eval_device,
                        "--session",
                        session_id,
                    ],
                    env=env_local,
                    dry_run=False,
                )

                eval_json = _find_eval_json(PROJECT_ROOT / "results", session_id)
                eval_metrics = _read_eval_metrics(eval_json)
                server_metrics = _read_server_metrics(PROJECT_ROOT / "results" / session_id)
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = str(exc)
            print(f"[MATRIX][ERROR] {run_id}: {error}")
        finally:
            if cleanup_cmd:
                subprocess.run(cleanup_cmd, cwd=PROJECT_ROOT, env=env_backend, check=False)

        ended_at = datetime.now().isoformat(timespec="seconds")
        duration_sec = round(time.time() - start_ts, 2)
        row = {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "scenario_desc": scenario_desc,
            "seed": seed,
            "session_id": session_id,
            "status": status,
            "error": error,
            "config_path": str(run_config_path),
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_sec": duration_sec,
        }
        row.update(eval_metrics)
        row.update(server_metrics)
        rows.append(row)
        _write_summary(_merge_rows_by_run_id(existing_rows, rows), summary_csv)
        print(f"[MATRIX] run done | status={status} | duration={duration_sec}s")
        print(f"[MATRIX] summary updated: {summary_csv}")

    ok_count = sum(1 for r in rows if r.get("status") == "ok")
    print(f"\n[MATRIX] completed runs={len(rows)} ok={ok_count} failed={len(rows) - ok_count}")
    print(f"[MATRIX] summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# test auto deploy

