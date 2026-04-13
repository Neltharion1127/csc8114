#!/usr/bin/env python3
"""
classify_client_weights.py
==========================
讀取 pi_bestweights 下每個 best_client_*.pth 的內容，
從裡面的 config_ref.config_path 或 config_snapshot.training.seed
辨別 seed，並移入對應的 <SCENARIO>_<SEED>/ 子目錄。

這是最準確的分類方式，不依賴日期或檔名猜測。

Pi 初始結構：
  pi_bestweights/<SESSION>/<SCENARIO>/
    best_client_X_round_Y_model_YYYYMMDD.pth   ← seed42 和 seed52 混在一起
    periodic/client_X_round_Y.pth              ← 不處理

整理後：
  pi_bestweights/<SESSION>/<SCENARIO>_seed42/
    best_client_X_round_Y_model_xxx.pth
  pi_bestweights/<SESSION>/<SCENARIO>_seed52/
    best_client_X_round_Y_model_xxx.pth

用法：
  python3 scripts/classify_client_weights.py \
      --session 2026-04-09_08-11-48 \
      --pi-root pi_bestweights \
      [--dry-run]
"""

import argparse
import os
import re
import shutil

import torch

SCENARIO_DIRS = [f"{i:02d}" for i in range(1, 15)]


def extract_seed_from_pth(pth_path: str) -> str | None:
    """
    讀取 .pth 檔案，從以下欄位依序嘗試取得 seed：
      1. ckpt['config_ref']['config_path']  -> 找 'seedXX'
      2. ckpt['config_snapshot']['training']['seed']
      3. ckpt['config']['seed']  (舊格式)
    """
    try:
        ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [WARN] 無法讀取 {os.path.basename(pth_path)}: {e}")
        return None

    # 方法 1: config_ref.config_path 裡的 seedXX (最可靠)
    config_path = ""
    if isinstance(ckpt.get("config_ref"), dict):
        config_path = ckpt["config_ref"].get("config_path", "")
    match = re.search(r"seed(\d+)", config_path)
    if match:
        return f"seed{match.group(1)}"

    # 方法 2: config_snapshot.training.seed 數字
    snapshot = ckpt.get("config_snapshot") or ckpt.get("config_minimal") or {}
    seed_num = snapshot.get("training", {}).get("seed")
    if seed_num is not None:
        return f"seed{int(seed_num)}"

    # 方法 3: 頂層 config.seed (舊格式備用)
    seed_num = ckpt.get("config", {}).get("seed")
    if seed_num is not None:
        return f"seed{int(seed_num)}"

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="依 seed 分類 pi_bestweights 的 client 權重")
    parser.add_argument("--session",  default="2026-04-09_08-11-48")
    parser.add_argument("--pi-root",  default="bestweights/pi_bestweights", help="Pi 權重的根目錄")
    parser.add_argument("--dry-run",  action="store_true", help="只顯示，不移動")
    args = parser.parse_args()

    session_dir = os.path.join(args.pi_root, args.session)
    if not os.path.isdir(session_dir):
        print(f"[ERROR] 找不到目錄：{session_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Session   : {args.session}")
    print(f"Pi root   : {session_dir}")
    print(f"Mode      : {'DRY-RUN' if args.dry_run else '實際執行'}")
    print(f"{'='*60}")

    total_moved   = 0
    total_unknown = 0
    total_skip    = 0

    for scenario in SCENARIO_DIRS:
        scenario_dir = os.path.join(session_dir, scenario)
        if not os.path.isdir(scenario_dir):
            continue

        best_files = sorted([
            f for f in os.listdir(scenario_dir)
            if f.startswith("best_client_") and f.endswith(".pth")
        ])

        if not best_files:
            continue

        print(f"\n📂 {scenario}/  ({len(best_files)} 個 best_client weights)")

        for fname in best_files:
            fpath = os.path.join(scenario_dir, fname)
            print(f"  🔍 讀取 {fname} ...", end=" ", flush=True)

            seed = extract_seed_from_pth(fpath)

            if seed is None:
                print(f"❌ 無法判斷 seed")
                total_unknown += 1
                continue

            dst_dir  = os.path.join(session_dir, f"{scenario}_{seed}")
            dst_path = os.path.join(dst_dir, fname)

            if os.path.exists(dst_path):
                print(f"⏭  已存在，跳過")
                total_skip += 1
                continue

            print(f"→ {scenario}_{seed}/")

            if not args.dry_run:
                os.makedirs(dst_dir, exist_ok=True)
                shutil.move(fpath, dst_path)

            total_moved += 1

    print(f"\n{'='*60}")
    print(f"✅ {'模擬' if args.dry_run else ''}完成")
    print(f"   移動 : {total_moved} 個")
    print(f"   跳過 : {total_skip} 個 (已存在)")
    print(f"   未知 : {total_unknown} 個 (請手動檢查)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
