import os
# src/common/config_loader.py
import yaml
from pathlib import Path
from typing import Any, Dict

DOCKER_CONFIG_DIR = Path("/app/config")
LOCAL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

CONFIG_DIR = DOCKER_CONFIG_DIR if DOCKER_CONFIG_DIR.exists() else LOCAL_CONFIG_DIR


def load_yaml(filename: str) -> Dict[str, Any]:
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _apply_env_overrides(conf: Dict[str, Any]) -> Dict[str, Any]:
    """
    Allow container-specific overrides (e.g., Airflow hostnames)
    without duplicating config files per runtime.
    """
    minio_cfg = conf.get("minio", {})
    minio_overrides = {
        "endpoint": os.getenv("MINIO_ENDPOINT"),
        "access_key": os.getenv("MINIO_ACCESS_KEY"),
        "secret_key": os.getenv("MINIO_SECRET_KEY"),
        "bucket": os.getenv("MINIO_BUCKET"),
        "secure": os.getenv("MINIO_SECURE"),
    }
    for key, value in minio_overrides.items():
        if value is None or value == "":
            continue
        if key == "secure":
            minio_cfg[key] = str(value).lower() in {"1", "true", "yes"}
        else:
            minio_cfg[key] = value

    kafka_cfg = conf.get("kafka", {})
    bootstrap_override = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    if bootstrap_override:
        kafka_cfg["bootstrap_servers"] = bootstrap_override

    return conf
    

def load_base_config() -> Dict[str, Any]:
    conf = load_yaml("base_config.yml")
    return _apply_env_overrides(conf)


def load_ingestion_file_config() -> Dict[str, Any]:
    return load_yaml("ingestion_file.yml")

