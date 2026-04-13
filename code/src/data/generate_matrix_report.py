import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Inject project root to sys.path so direct execution works without ModuleNotFoundError
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.shared.common import get_config

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, help="Session ID (e.g. 2026-04-07_11-00-15)")
    parser.add_argument("--device", default="cpu", help="cpu, mps, or cuda")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    session_dir = project_root / "results" / args.session
    if not session_dir.is_dir():
        print(f"[ERROR] Session directory not found: {session_dir}")
        return

    # Check for scenario subdirectories like "scenario_1", "01_seed42", etc.
    sub_scenarios = sorted([d for d in session_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    # If no scenario folders, treat the session itself as one
    scenarios_to_process = sub_scenarios if sub_scenarios else [session_dir]

    print(f"\n\U0001f4ca  GENERATING MATRIX REPORT for Session: {args.session}")
    print(f"   Mode: {'Multi-Scenario Matrix' if sub_scenarios else 'Single Session'}")
    print(f"   Found {len(scenarios_to_process)} target(s). Building profiles...\n")

    summary_list = []
    
    for idx, scen_dir in enumerate(scenarios_to_process):
        # Determine unique session path for evaluation
        rel_path = scen_dir.relative_to(project_root / "results")
        scen_name = scen_dir.name
        
        print(f"\r  [\u231b] Processing {idx+1}/{len(scenarios_to_process)}: {scen_name}...", end="", flush=True)
        
        import json
        safe_scen_name = str(rel_path).replace("/", "_").replace("\\", "_")
        
        # Priority 1: Check in global 'reports/' directory (new location)
        reports_dir = project_root / "reports"
        json_path_global = reports_dir / f"evaluation_report_{safe_scen_name}.json"
        
        # Priority 2: Check in scenario directory (old location)
        json_path_local = scen_dir / f"evaluation_report_{safe_scen_name}.json"
        
        target_json = None
        if json_path_global.exists():
            target_json = json_path_global
        elif json_path_local.exists():
            target_json = json_path_local
            
        if not target_json:
            # Fallback scan in reports/ if exact name match fails (handle slight diff in path joining)
            possible_files = list(reports_dir.glob(f"evaluation_report_*_{scen_name}.json"))
            if possible_files:
                target_json = possible_files[0]

        if not target_json:
            print(f"\n    [WARNING] No evaluation JSON found for {scen_name}. Searched in:\n      - {json_path_global}\n      - {json_path_local}\n    Skip.")
            continue
            
        try:
            with open(target_json, "r") as f:
                data = json.load(f)
            
            # Use Client 1 as the representative for the scenario
            clients_data = data.get("clients", [])
            if not clients_data:
                print(f"\n    [WARNING] No client data found in JSON for {scen_name}. Skip.")
                continue
                
            all_station_details = []
            sum_mse = 0.0
            sum_f1 = 0.0
            sum_acc = 0.0
            valid_clients = 0
            
            for client in clients_data:
                details = client.get("monthly_details", [])
                all_station_details.extend(details)
                
                # Check for metrics across clients
                if "mse" in client:
                    sum_mse += client["mse"]
                    sum_f1 += client.get("f1", 0.0)
                    sum_acc += client.get("accuracy", 0.0)
                    valid_clients += 1
            
            if not all_station_details or valid_clients == 0:
                print(f"\n    [WARNING] No station details found for {scen_name}. Skip.")
                continue
                
            avg_mse = sum_mse / valid_clients
            avg_f1 = sum_f1 / valid_clients
            avg_acc = sum_acc / valid_clients
                
        except Exception as e:
            print(f"\n    [ERROR] Failed to parse JSON for {scen_name}: {e}")
            continue

        df_monthly = pd.DataFrame(all_station_details)
        
        # Map numeric months to names
        month_map = {
            "01": "January", "02": "February", "03": "March", "04": "April",
            "05": "May", "06": "June", "07": "July", "08": "August",
            "09": "September", "10": "October", "11": "November", "12": "December"
        }
        if "Month" in df_monthly.columns:
            df_monthly["Month"] = df_monthly["Month"].map(lambda x: month_map.get(x, x))

        monthly_profile = df_monthly.groupby("Month").agg({
            "MSE": "mean", "Acc": "mean", "F1": "mean", "Samples": "sum"
        }).reset_index()

        scen_summary = {
            "Scenario": scen_name,
            "MSE_Avg": round(avg_mse, 6),
            "F1_Avg": round(avg_f1, 4),
            "Acc_Avg": round(avg_acc, 4),
            "Latency_ms": 0.0,
            "Traffic_KB": 0.0,
            "Station_Profiles": monthly_profile.to_dict(orient="records")
        }

        # Efficiency metrics
        logs = list(scen_dir.glob("training_log_client*.csv"))
        if logs:
            try:
                df_logs = pd.concat([pd.read_csv(f) for f in logs])
                # Check for either Profiler column format or simple log format
                if "LatencyMs" in df_logs.columns:
                    scen_summary["Latency_ms"] = round(df_logs["LatencyMs"].mean(), 1)
                else:
                    lat_cols = [c for c in ["time_total", "time_forward", "time_backward"] if c in df_logs.columns]
                    if lat_cols:
                        scen_summary["Latency_ms"] = round(df_logs[lat_cols].sum(axis=1).mean() * 1000, 1)
                        
                if "PayloadBytes" in df_logs.columns:
                    scen_summary["Traffic_KB"] = round(df_logs["PayloadBytes"].mean() / 1024, 1)
                elif "payload_size" in df_logs.columns:
                    scen_summary["Traffic_KB"] = round(df_logs["payload_size"].mean() / 1024, 1)
            except: pass

        summary_list.append(scen_summary)

    print(f"\n\n\u2705 ALL DATA PROCESSED.\n")

    # 1. REPORT: Monthly Breakdown Export
    station_rows = []
    for s in summary_list:
        print(f"\n📁  Scenario: {s['Scenario']}")
        print(f"{'Month':<15} | {'F1':<8} | {'Acc':<8} | {'MSE':<10} | {'Latency':<8} | {'Traffic':<8}")
        print("-" * 75)
        
        # Define month order for sorting
        month_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        sorted_profiles = sorted(s['Station_Profiles'], 
                                 key=lambda x: month_order.index(x['Month']) if x['Month'] in month_order else 99)
        
        for m in sorted_profiles:
            st_name = m['Month']
            
            # Print console row
            print(f"{st_name:<15} | {m['F1']:<8.4f} | {m['Acc']:<8.4f} | {m['MSE']:<10.6f} | {s['Latency_ms']:<8.1f} | {s['Traffic_KB']:<8.1f}")
            
            # Prepare row for CSV
            station_rows.append({
                "Scenario": s['Scenario'],
                "Station": st_name,
                "F1": round(m['F1'], 4),
                "Accuracy": round(m['Acc'], 4),
                "MSE": round(m['MSE'], 6),
                "Latency_ms": s['Latency_ms'],
                "Traffic_KB": s['Traffic_KB']
            })

    # Save Flattened Station CSV
    if station_rows:
        df_monthly = pd.DataFrame(station_rows)
        monthly_csv_path = project_root / "results" / args.session / f"Matrix_Station_Details_{args.session}.csv"
        df_monthly.to_csv(monthly_csv_path, index=False)
        print(f"\n📊  Report B (Scenarios x Stations) saved to: {monthly_csv_path}")
        
        # ALSO save to reports/ directory
        reports_monthly_path = project_root / "reports" / f"Matrix_Station_Details_{args.session}.csv"
        df_monthly.to_csv(reports_monthly_path, index=False)
        print(f"📁  Copy saved to: {reports_monthly_path}")

    # 2. REPORT: PK Matrix (Report A)
    print("\n\n" + "="*80)
    print("  FINAL CONSOLIDATED PROJECT PK MATRIX (Global Summary)")
    print("="*80)
    if summary_list:
        df_global = pd.DataFrame(summary_list).drop(columns=["Station_Profiles"])
        print(df_global.to_string(index=False))
        global_csv_path = project_root / "results" / args.session / f"Matrix_Global_Summary_{args.session}.csv"
        df_global.to_csv(global_csv_path, index=False)
        print(f"\n\U0001f4be  Report A (Global Summary) saved to: {global_csv_path}")
        
        # ALSO save to reports/ directory
        reports_global_path = project_root / "reports" / f"Matrix_Global_Summary_{args.session}.csv"
        df_global.to_csv(reports_global_path, index=False)
        print(f"📁  Copy saved to: {reports_global_path}")

if __name__ == "__main__":
    main()
