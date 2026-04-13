import copy
import hashlib
import json

from src.shared.common import cfg, get_config_path, get_nested

_DEFAULT_POLICY = "minimal"
_VALID_POLICIES = {"none", "minimal", "full"}
_MINIMAL_PATHS = (
    ("compression", "mode"),
    ("compression", "topk_ratio"),
    ("data", "feature_cols"),
    ("data", "processed_dir"),
    ("data", "train_end"),
    ("data", "val_end"),
    ("data_download", "end_date"),
    ("federated", "num_clients"),
    ("federated", "rho"),
    ("model", "horizon"),
    ("model", "hidden_size"),
    ("model", "input_size"),
    ("model", "lstm_dropout"),
    ("model", "num_layers"),
    ("model", "seq_len"),
    ("model", "server_head_dropout"),
    ("model", "server_head_width"),
    ("profiler", "enabled"),
    ("scheduler", "enabled"),
    ("training", "checkpoint_interval"),
    ("training", "classification_loss_type"),
    ("training", "classification_positive_weight"),
    ("training", "classification_loss_weight"),
    ("training", "focal_alpha"),
    ("training", "focal_gamma"),
    ("training", "num_rounds"),
    ("training", "rain_threshold_mm"),
    ("training", "rain_probability_threshold"),
    ("training", "regression_loss_weight"),
    ("training", "seed"),
    ("training", "target_transform"),
)


def _set_nested(target: dict, path: tuple[str, ...], value) -> None:
    cursor = target
    for key in path[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[path[-1]] = value


def _stable_json_payload(source: dict) -> bytes:
    return json.dumps(
        source,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")


def resolve_config_snapshot_policy(config: dict | None = None) -> str:
    source = cfg if config is None else config
    raw_policy = str(get_nested(source, ("artifacts", "config_snapshot_policy"), _DEFAULT_POLICY)).lower().strip()
    if raw_policy not in _VALID_POLICIES:
        return _DEFAULT_POLICY
    return raw_policy


def config_sha256(config: dict | None = None) -> str:
    source = cfg if config is None else config
    return hashlib.sha256(_stable_json_payload(source)).hexdigest()


def build_config_ref(config: dict | None = None) -> dict[str, str]:
    source = cfg if config is None else config
    return {
        "config_path": get_config_path(),
        "config_sha256": config_sha256(source),
    }


def build_minimal_config_snapshot(config: dict | None = None) -> dict:
    source = cfg if config is None else config
    snapshot: dict = {}
    for path in _MINIMAL_PATHS:
        value = get_nested(source, path, None)
        if value is not None:
            _set_nested(snapshot, path, value)
    return snapshot


def build_config_snapshot(config: dict | None = None) -> tuple[dict | None, str]:
    source = cfg if config is None else config
    policy = resolve_config_snapshot_policy(source)
    if policy == "none":
        return None, policy
    if policy == "full":
        return copy.deepcopy(source), policy
    return build_minimal_config_snapshot(source), policy
