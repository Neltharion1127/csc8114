#!/usr/bin/env python3
"""
rename_seed_dirs.py
===================
將每個 scenario 資料夾內的 seedXX 子目錄加上 scenario 前綴。

例如：
  01/seed42  ->  01/01_seed42
  01/seed52  ->  01/01_seed52
  14/seed42  ->  14/14_seed42

用法：
  python3 scripts/rename_seed_dirs.py \
      --session results/pi_results/2026-04-09_08-11-48 \
      [--dry-run]
"""

import argparse
import os

SCENARIO_DIRS = [f"{i:02d}" for i in range(1, 15)]   # 01 ~ 14


def main() -> None:
    parser = argparse.ArgumentParser(description="將 seedXX 改名為 <scenario>_seedXX")
    parser.add_argument(
        "--session",
        default="results/pi_results/2026-04-09_08-11-48",
        help="Session 根目錄路徑",
    )
    parser.add_argument("--dry-run", action="store_true", help="只顯示，不真的改名")
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

        # 找出該 scenario 下的 seedXX 資料夾 (不含已加前綴的)
        seed_dirs = sorted([
            d for d in os.listdir(scenario_dir)
            if os.path.isdir(os.path.join(scenario_dir, d))
            and d.startswith("seed")          # 只改 seedXX（未加前綴的）
            and not d.startswith(scenario)    # 跳過已命名為 01_seed42 的
        ])

        if not seed_dirs:
            continue

        print(f"\n📂 {scenario}/")
        for seed_dir in seed_dirs:
            old_path = os.path.join(scenario_dir, seed_dir)
            new_name = f"{scenario}_{seed_dir}"           # 例如 01_seed42
            new_path = os.path.join(scenario_dir, new_name)

            if args.dry_run:
                print(f"  [DRY] {seed_dir}  ->  {new_name}")
            else:
                os.rename(old_path, new_path)
                print(f"  [RN]  {seed_dir}  ->  {new_name}")
            total += 1

    print(f"\n{'='*60}")
    print(f"✅ {'模擬' if args.dry_run else ''}完成，共重新命名 {total} 個資料夾。")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
