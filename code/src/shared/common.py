import os
import sys
import yaml

DEFAULT_FEATURE_COLS = ["Temperature", "Humidity", "Pressure", "Wind Speed", "Rain"]

# Resolve project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

CONFIG_PATH = os.path.join(project_root, "config.yaml")
try:
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f) or {}
except Exception:
    cfg = {}

def get_config():
    """Return the loaded configuration dict."""
    return cfg


def reload_config():
    """Reload configuration from disk into the module-level `cfg`."""
    global cfg
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    return cfg


def get_nested(config: dict | None, keys: tuple[str, ...], default=None):
    """Safely fetch a nested key path from a dict-like config."""
    current = config if isinstance(config, dict) else {}
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def feature_cols_from_cfg(config: dict | None = None) -> list[str]:
    """
    Resolve feature columns with backward-compatible fallback:
      1) data.feature_cols (preferred)
      2) data_download.feature_cols (legacy)
      3) hardcoded defaults
    """
    source = cfg if config is None else config
    for key_path in (("data", "feature_cols"), ("data_download", "feature_cols")):
        cols = get_nested(source, key_path, None)
        if isinstance(cols, list) and cols:
            return [str(col) for col in cols]
    return DEFAULT_FEATURE_COLS.copy()
