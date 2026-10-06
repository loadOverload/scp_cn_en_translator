"""Configuration loading, path resolution and environment setup.

Every script in this project starts with:

    from src.utils.config import load_config
    cfg = load_config("configs/default.yaml")

The loader:
  * reads a YAML file
  * recursively follows an optional ``defaults:`` key (base config)
  * deep-merges CLI overrides on top
  * resolves every path under ``paths`` / known file keys against project_root
  * points HuggingFace caches at a writable directory (the sandbox $HOME is
    read-only, and huggingface.co is unreachable -> HF_ENDPOINT is set to a
    reachable mirror).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import yaml

# ----------------------------------------------------------------------------
# YAML helpers
# ----------------------------------------------------------------------------


class Config(dict):
    """dict with attribute-style access and dotted keys.

    cfg.model.name_or_path  ==  cfg["model"]["name_or_path"]
    """

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - convenience only
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: Dict[str, Any] = self
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value


def _to_config(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return Config({k: _to_config(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config(v) for v in obj]
    return obj


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path, _seen: Iterable[Path] = ()) -> Dict[str, Any]:
    path = Path(path).resolve()
    if path in set(_seen):
        raise ValueError(f"circular 'defaults:' chain detected at {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"config {path} must contain a YAML mapping")

    defaults_ref = raw.pop("defaults", None)
    merged: Dict[str, Any] = {}
    if defaults_ref:
        # `defaults` is resolved relative to the config file that declares it
        candidates = [path.parent / str(defaults_ref), Path(str(defaults_ref))]
        base_path = next((c for c in candidates if c.exists()), None)
        if base_path is None:
            raise FileNotFoundError(f"defaults '{defaults_ref}' not found (from {path})")
        merged = _read_yaml(base_path, list(_seen) + [path])
    return _deep_merge(merged, dict(raw))


# ----------------------------------------------------------------------------
# Path resolution
# ----------------------------------------------------------------------------

# keys whose string values are filesystem paths
_PATH_KEYS = {
    "db_path",
    "export_jsonl",
    "pairs_file",
    "train",
    "val",
    "test",
    "chat_train",
    "chat_val",
    "output_dir",
    "log_dir",
    "hf_home",
    "splits_dir",
    "cleaned_dir",
    "raw_dir",
    "raw_en_dir",
    "raw_zh_dir",
    "samples_dir",
    "processed_dir",
    "qa_dir",
    "logo",
}


def _resolve_paths(node: Any, root: Path, key: str | None = None) -> Any:
    if isinstance(node, Mapping):
        return Config({k: _resolve_paths(v, root, k) for k, v in node.items()})
    if isinstance(node, list):
        return [_resolve_paths(v, root, key) for v in node]
    if isinstance(node, str) and key in _PATH_KEYS and node:
        p = Path(node).expanduser()          # allow ~/... in the config
        if not p.is_absolute():
            p = root / p
        return str(p)
    return node


def load_config(
    config_path: str | Path = "configs/default.yaml",
    overrides: Mapping[str, Any] | Sequence[str] | None = None,
    project_root: str | Path | None = None,
) -> Config:
    """Load a YAML config, merge overrides, resolve paths, set HF env vars."""
    config_path = Path(config_path)
    if not config_path.is_absolute() and not config_path.exists():
        # tolerate being called from scripts/
        alt = Path(__file__).resolve().parents[2] / config_path
        if alt.exists():
            config_path = alt

    raw = _read_yaml(config_path)

    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
    else:
        configured = Path(str(raw.get("project_root", "."))).expanduser()
        if configured.is_absolute():
            root = configured
        else:
            # project_root is relative to the config file's parent's parent
            root = (config_path.resolve().parent.parent / configured).resolve()
    raw["project_root"] = str(root)

    cfg = _to_config(raw)

    if overrides:
        override_map = _parse_overrides(overrides)
        for dotted, value in override_map.items():
            cfg.set_path(dotted, value)

    resolved = _resolve_paths(dict(cfg), root)
    out = Config(resolved)
    out["project_root"] = str(root)
    setup_environment(out)
    return out


def _parse_overrides(overrides: Mapping[str, Any] | Sequence[str]) -> Dict[str, Any]:
    if isinstance(overrides, Mapping):
        return dict(overrides)
    result: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        key, _, value = item.partition("=")
        result[key.strip()] = yaml.safe_load(value)
    return result


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------


def setup_environment(cfg: Mapping[str, Any]) -> None:
    """Point caches at writable locations and pick a reachable HF endpoint.

    Must run BEFORE `import transformers` / `datasets` so the caches are honoured.
    """
    hf_home = cfg.get("paths", {}).get("hf_home") if isinstance(cfg.get("paths"), Mapping) else None
    if hf_home:
        hf_home = Path(str(hf_home))
        hf_home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_home)
        os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
        # keep temp files off the read-only home
        os.environ.setdefault("TMPDIR", str(hf_home.parent / "tmp"))
        Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    # hf-mirror does not implement the Xet protocol; leaving it enabled makes
    # downloads fail with HTTP 401 from cas-server.xethub.hf.co.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    endpoint = (cfg.get("hf") or {}).get("endpoint") if isinstance(cfg.get("hf"), Mapping) else None
    if endpoint:
        os.environ["HF_ENDPOINT"] = str(endpoint)
    if (cfg.get("hf") or {}).get("local_files_only") if isinstance(cfg.get("hf"), Mapping) else False:
        os.environ["HF_HUB_OFFLINE"] = "1"

    # torch / triton scratch
    root = Path(str(cfg.get("project_root", ".")))
    os.environ.setdefault("TORCH_HOME", str(root / "data" / "hf_cache" / "torch"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(root / "data" / "hf_cache" / "triton"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def ensure_dirs(cfg: Mapping[str, Any]) -> None:
    """Create every directory referenced by the config."""
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths"), Mapping) else {}
    for value in paths.values():
        if isinstance(value, str) and value:
            p = Path(value)
            if p.suffix == "":  # looks like a directory
                p.mkdir(parents=True, exist_ok=True)
    for section in ("evaluation",):
        node = cfg.get(section)
        if isinstance(node, Mapping):
            for key in ("db_path", "export_jsonl", "output_dir"):
                if isinstance(node.get(key), str):
                    Path(node[key]).parent.mkdir(parents=True, exist_ok=True)


def add_common_args(parser) -> None:  # pragma: no cover - argparse sugar
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set training.max_steps=10 (repeatable)",
    )
