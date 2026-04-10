import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_root(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _list_sessions(root: Path) -> list[str]:
    if not root.exists():
        return []
        
    # Find all directories containing a server model (recursive)
    found_dirs = set()
    for p in root.glob("**/server_head_round_*.pth"):
        # Get path relative to root
        try:
            rel_dir = p.parent.relative_to(root)
            if str(rel_dir) == ".":
                # If weights are in root (flat layout), we don't treat root as a session name 
                # unless we want to support it specifically. 
                # For this project, sessions are subdirs.
                pass
            else:
                found_dirs.add(str(rel_dir))
        except ValueError:
            continue
            
    if not found_dirs:
        # Fallback to legacy behavior for backward compatibility
        found_dirs = [
            p.name for p in root.iterdir() 
            if p.is_dir() and (p.name.startswith("20") or "seed" in p.name)
        ]
        
    return sorted(list(found_dirs))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch wrapper for src.data.run_evaluation across many sessions."
    )
    parser.add_argument(
        "--sessions-root",
        default="bestweights",
        help="Directory that contains session folders (default: bestweights).",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated session IDs to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of sessions to run (0 means all).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Evaluation device passed through to run_evaluation (cpu/mps/cuda).",
    )
    parser.add_argument(
        "--force-prob-threshold",
        type=float,
        default=None,
        help="Optional fixed probability threshold in [0,1] for all sessions.",
    )
    parser.add_argument(
        "--report-tag",
        default="",
        help="Optional report tag suffix passed through to run_evaluation.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining sessions even if one evaluation fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    args = parser.parse_args()

    if args.force_prob_threshold is not None and not (0.0 <= float(args.force_prob_threshold) <= 1.0):
        raise ValueError("--force-prob-threshold must be in [0, 1].")

    sessions_root = _resolve_root(args.sessions_root)
    sessions = _list_sessions(sessions_root)
    only_ids = {s.strip() for s in args.only.split(",") if s.strip()}
    if only_ids:
        sessions = [s for s in sessions if s in only_ids]
    if args.limit > 0:
        sessions = sessions[: args.limit]

    if not sessions:
        raise FileNotFoundError(f"No sessions found under {sessions_root}")

    print(f"[BATCH-EVAL] sessions_root={sessions_root}")
    print(f"[BATCH-EVAL] total_sessions={len(sessions)}")

    bw_dir = PROJECT_ROOT / "bestweights"
    try:
        rel_prefix = sessions_root.relative_to(bw_dir)
    except ValueError:
        rel_prefix = Path("")

    failures: list[str] = []
    for idx, session_id in enumerate(sessions, start=1):
        # e.g., if rel_prefix is '2026-04-09_08-11-48', combine it with '01_seed42'
        full_session_path = str(rel_prefix / session_id) if str(rel_prefix) != "." else session_id

        cmd = [
            sys.executable,
            "-m",
            "src.data.run_evaluation",
            "--device",
            str(args.device),
            "--session",
            full_session_path,
        ]
        if args.force_prob_threshold is not None:
            cmd.extend(["--force-prob-threshold", str(args.force_prob_threshold)])
        if args.report_tag:
            cmd.extend(["--report-tag", str(args.report_tag)])

        printable = " ".join(cmd)
        print(f"\n[BATCH-EVAL] [{idx}/{len(sessions)}] {session_id}")
        print(f"[BATCH-EVAL][CMD] {printable}")

        if args.dry_run:
            continue

        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        if result.returncode != 0:
            failures.append(session_id)
            print(f"[BATCH-EVAL][ERROR] session={session_id} returncode={result.returncode}")
            if not args.continue_on_error:
                break

    if failures:
        print(f"\n[BATCH-EVAL] completed_with_failures={len(failures)} failed_sessions={failures}")
        return 1

    print("\n[BATCH-EVAL] all sessions completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
