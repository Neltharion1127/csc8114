#!/usr/bin/env python3
"""
flatten_for_report.py
=====================
將 pi_results 下多層的目錄結構「打平」成 matrix-report 能直接讀取的格式。

來源結構 (pi_results 整理後)：
  results/pi_results/<SESSION>/
    01/
      01_seed42/
        training_log_client1_xxx.csv
        training_log_client1_xxx_meta.json
        ...
      01_seed52/
    02/
      02_seed42/
      ...

目標結構 (matrix-report 需要)：
  results/<SESSION>/
    01_seed42/
      training_log_client1_xxx.csv
      ...
    01_seed52/
    02_seed42/
    ...

預設行為：複製 (保留 pi_results 原始檔案)。
加上 --move 改為搬移。

用法：
  python3 scripts/flatten_for_report.py \
      --session 2026-04-09_08-11-48 \
      [--move] \
      [--dry-run]
"""

import argparse
import os
import shutil

SCENARIO_DIRS = [f"{i:02d}" for i in range(1, 15)]   # 01 ~ 14


def main() -> None:
    parser = argparse.ArgumentParser(description="打平 pi_results 目錄給 matrix-report 使用")
    parser.add_argument(
        "--session",
        default="2026-04-09_08-11-48",
        help="Session ID (例如: 2026-04-09_08-11-48)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="搬移 (預設為複製)",
    )
    parser.add_argument("--dry-run", action="store_true", help="只顯示，不實際操作")
    args = parser.parse_args()

    src_root = os.path.join("results", "pi_results", args.session)
    dst_root = os.path.join("results", args.session)
    action   = "移動" if args.move else "複製"

    if not os.path.isdir(src_root):
        print(f"[ERROR] 來源目錄找不到：{src_root}")
        return

    print(f"\n{'='*60}")
    print(f"Session   : {args.session}")
    print(f"來源       : {src_root}")
    print(f"目標       : {dst_root}")
    print(f"動作       : {action}")
    print(f"Mode      : {'DRY-RUN' if args.dry_run else '實際執行'}")
    print(f"{'='*60}")

    total = 0
    skipped = 0

    for scenario in SCENARIO_DIRS:
        scenario_dir = os.path.join(src_root, scenario)
        if not os.path.isdir(scenario_dir):
            continue

        # 找出 XX_seedYY 子目錄
        seed_dirs = sorted([
            d for d in os.listdir(scenario_dir)
            if os.path.isdir(os.path.join(scenario_dir, d))
            and "seed" in d
        ])

        if not seed_dirs:
            print(f"\n[SKIP] {scenario}/ 沒有 seedXX 子目錄")
            continue

        print(f"\n📂 {scenario}/")
        for seed_dir in seed_dirs:
            src_seed_dir = os.path.join(scenario_dir, seed_dir)
            dst_seed_dir = os.path.join(dst_root, seed_dir)

            # 如果目標已存在，警告
            if os.path.exists(dst_seed_dir):
                print(f"  [WARN] 目標已存在，跳過：{dst_seed_dir}")
                skipped += 1
                continue

            if args.dry_run:
                print(f"  [DRY] {action}: {src_seed_dir}")
                print(f"          -> {dst_seed_dir}")
            else:
                os.makedirs(dst_root, exist_ok=True)
                if args.move:
                    shutil.move(src_seed_dir, dst_seed_dir)
                else:
                    shutil.copytree(src_seed_dir, dst_seed_dir)
                print(f"  [OK]  {seed_dir}  ->  results/{args.session}/{seed_dir}")

            total += 1

    print(f"\n{'='*60}")
    print(f"✅ {'模擬' if args.dry_run else ''}完成 | {action} {total} 個場景資料夾 | 跳過 {skipped} 個")
    if not args.dry_run and total > 0:
        print(f"\n📌 現在可以執行：")
        print(f"   make matrix-report SESSION={args.session}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
