#!/usr/bin/env python3
"""
organize_by_seed.py
===================
讀取 results/pi_results/<session>/<01~14>/ 內帶日期的 _meta.json，
依照 seed 將 csv + meta.json 一起移入 seedXX/ 子目錄。

每個 meta.json 對應一個同名的 csv：
  training_log_client1_20260409_100300_meta.json
  training_log_client1_20260409_100300.csv
  -> 移入 seed42/

seed 來源：meta.json 內的 config_ref.config_path (優先)
          或 config_minimal.training.seed (備用)

用法：
  python scripts/organize_by_seed.py \
      --session results/pi_results/2026-04-09_08-11-48 \
      [--dry-run]
"""

import argparse
import json
import os
import re
import shutil

SCENARIO_DIRS = [f"{i:02d}" for i in range(1, 15)]   # 01 ~ 14


def extract_seed(meta_path: str) -> str | None:
    """從 _meta.json 讀出 seed 字串，例如 'seed42'。"""
    try:
        with open(meta_path) as f:
            meta = json.load(f)

        # 方法 A：config_ref.config_path 裡的 seedXX (最可靠)
        config_path = meta.get("config_ref", {}).get("config_path", "")
        match = re.search(r"seed(\d+)", config_path)
        if match:
            return f"seed{match.group(1)}"

        # 方法 B：config_minimal.training.seed 的數字
        seed_num = meta.get("config_minimal", {}).get("training", {}).get("seed")
        if seed_num is not None:
            return f"seed{int(seed_num)}"

    except Exception as e:
        print(f"  [WARN] 無法讀取 {meta_path}: {e}")

    return None


def move_file(src: str, dst: str, dry_run: bool) -> None:
    if dry_run:
        print(f"    [DRY] mv {os.path.basename(src)}  ->  {os.path.basename(dst)}")
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        print(f"    [MV]  {os.path.basename(src)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="依照 seed 整理帶日期的結果檔案")
    parser.add_argument(
        "--session",
        default="results/pi_results/2026-04-09_08-11-48",
        help="Session 根目錄路徑",
    )
    parser.add_argument("--dry-run", action="store_true", help="只顯示計畫，不移動")
    args = parser.parse_args()

    session_root = args.session
    if not os.path.isdir(session_root):
        print(f"[ERROR] 找不到 session 目錄：{session_root}")
        return

    print(f"\n{'='*60}")
    print(f"Session: {session_root}")
    print(f"Mode:    {'DRY-RUN' if args.dry_run else '實際執行'}")
    print(f"{'='*60}")

    total_moved = 0
    total_skipped = 0

    for scenario in SCENARIO_DIRS:
        scenario_dir = os.path.join(session_root, scenario)
        if not os.path.isdir(scenario_dir):
            continue

        # 只處理帶日期的 meta.json (排除 progress)
        meta_files = sorted([
            f for f in os.listdir(scenario_dir)
            if f.endswith("_meta.json") and "_progress" not in f
        ])

        if not meta_files:
            continue

        print(f"\n📂 {scenario}/  ({len(meta_files)} 個 meta.json)")

        for meta_fname in meta_files:
            meta_path = os.path.join(scenario_dir, meta_fname)
            seed = extract_seed(meta_path)

            if seed is None:
                print(f"  [SKIP] 無法判斷 seed：{meta_fname}")
                total_skipped += 1
                continue

            csv_fname = meta_fname.replace("_meta.json", ".csv")
            csv_path  = os.path.join(scenario_dir, csv_fname)

            seed_dir  = os.path.join(scenario_dir, seed)
            print(f"\n  📌 {seed}/  ({meta_fname.split('_meta')[0]})")

            move_file(meta_path, os.path.join(seed_dir, meta_fname), args.dry_run)
            total_moved += 1

            if os.path.exists(csv_path):
                move_file(csv_path, os.path.join(seed_dir, csv_fname), args.dry_run)
                total_moved += 1
            else:
                print(f"    [WARN] 對應 CSV 不存在：{csv_fname}")

    print(f"\n{'='*60}")
    print(f"✅ {'模擬' if args.dry_run else ''}完成 | 搬移 {total_moved} 個檔案 | 跳過 {total_skipped} 個")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
