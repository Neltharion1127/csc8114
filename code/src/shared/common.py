import os
import sys
import yaml

DEFAULT_FEATURE_COLS = ["Temperature", "Humidity", "Pressure", "Wind Speed", "Rain"]

# Resolve project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)


def _resolve_config_path() -> str:
    """Resolve config path with optional env override."""
    env_path = os.environ.get("FSL_CONFIG_PATH", "").strip()
    path = env_path if env_path else os.path.join(project_root, "config.yaml")
    if not os.path.isabs(path):
        path = os.path.join(project_root, path)
    return os.path.abspath(path)


def _load_config_or_raise(path: str) -> dict:
    """Load YAML config and fail fast on invalid/missing config."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping/dict: {path}")
    return loaded


CONFIG_PATH = _resolve_config_path()
cfg = _load_config_or_raise(CONFIG_PATH)

def get_config():
    """Return the loaded configuration dict."""
    return cfg


def reload_config():
    """Reload configuration from disk into the module-level `cfg`."""
    global cfg, CONFIG_PATH
    CONFIG_PATH = _resolve_config_path()
    cfg = _load_config_or_raise(CONFIG_PATH)
    return cfg


def get_config_path() -> str:
    """Return the active config path for this process."""
    return CONFIG_PATH


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
