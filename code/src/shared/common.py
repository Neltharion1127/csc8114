import os
import sys
import yaml

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
