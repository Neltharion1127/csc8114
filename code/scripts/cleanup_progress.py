#!/usr/bin/env python3
"""
cleanup_progress.py
===================
刪除 results/pi_results/<session>/<01~14>/ 內的所有 progress 暫存檔：
  - training_log_clientX_progress.csv
  - training_log_clientX_progress_meta.json

用法：
  python scripts/cleanup_progress.py \
      --session results/pi_results/2026-04-09_08-11-48 \
      [--dry-run]
"""

import argparse
import os

SCENARIO_DIRS = [f"{i:02d}" for i in range(1, 15)]   # 01 ~ 14


def main() -> None:
    parser = argparse.ArgumentParser(description="刪除 progress 暫存檔")
    parser.add_argument(
        "--session",
        default="results/pi_results/2026-04-09_08-11-48",
        help="Session 根目錄路徑",
    )
    parser.add_argument("--dry-run", action="store_true", help="只顯示，不刪除")
    args = parser.parse_args()

    session_root = args.session
    if not os.path.isdir(session_root):
        print(f"[ERROR] 找不到 session 目錄：{session_root}")
        return

    print(f"\n{'='*60}")
    print(f"Session: {session_root}")
    print(f"Mode:    {'DRY-RUN' if args.dry_run else '實際執行'}")
    print(f"{'='*60}")

    total = 0
    for scenario in SCENARIO_DIRS:
        scenario_dir = os.path.join(session_root, scenario)
        if not os.path.isdir(scenario_dir):
            continue

        progress_files = sorted([
            f for f in os.listdir(scenario_dir)
            if "_progress" in f
        ])

        if not progress_files:
            continue

        print(f"\n📂 {scenario}/  ({len(progress_files)} 個暫存檔)")
        for fname in progress_files:
            fpath = os.path.join(scenario_dir, fname)
            if args.dry_run:
                print(f"  [DRY] rm {fname}")
            else:
                os.remove(fpath)
                print(f"  [RM]  {fname}")
            total += 1

    print(f"\n{'='*60}")
    print(f"✅ {'模擬' if args.dry_run else ''}完成，共處理 {total} 個 progress 檔案。")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
