"""
Resolve and write a merged scenario config for native-run-single.

Usage:
    python -m src.shared.resolve_scenario_config <SCENARIO_ID>

Finds the scenario in matrix.yaml's experiment_matrix.scenarios, deep-merges
its overrides onto the base config (config.yaml), writes the result to
matrix_configs/, and prints the output path.  Prints an empty string if
SCENARIO_ID is empty or not found (caller falls back to the default config.yaml).
"""
import copy
import sys
import yaml
from datetime import datetime
from pathlib import Path


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def main() -> None:
    scenario_id = sys.argv[1].strip() if len(sys.argv) > 1 else ""
    if not scenario_id:
        print("")
        return

    here = Path(__file__).resolve().parent
    project_root = here.parent.parent
    config_path = project_root / "config.yaml"
    matrix_path = project_root / "matrix.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        root_cfg = yaml.safe_load(f) or {}

    with open(matrix_path, "r", encoding="utf-8") as f:
        matrix_raw = yaml.safe_load(f) or {}

    scenarios = matrix_raw.get("experiment_matrix", {}).get("scenarios", [])
    scenario = next(
        (s for s in scenarios if str(s.get("id", "")).strip() == scenario_id),
        None,
    )
    if scenario is None:
        print(
            f"[resolve_scenario_config] WARNING: scenario '{scenario_id}' not found "
            f"in experiment_matrix.scenarios — using default config.yaml",
            file=sys.stderr,
        )
        print("")
        return

    run_cfg = copy.deepcopy(root_cfg)
    overrides = scenario.get("overrides", {})
    if isinstance(overrides, dict) and overrides:
        _deep_merge(run_cfg, overrides)

    configs_dir = project_root / "results" / "matrix_configs"
    configs_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = configs_dir / f"{stamp}_{scenario_id}_native.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(run_cfg, f, sort_keys=False)

    print(str(out_path))


if __name__ == "__main__":
    main()
