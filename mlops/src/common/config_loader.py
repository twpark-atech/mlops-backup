# src/common/config_loader.py

from pathlib import Path
from typing import Any, Dict

import yaml

# 도커 컨테이너 기준 경로
DOCKER_CONFIG_DIR = Path("/app/config")
# 로컬(WSL)에서 프로젝트 루트/mlops 기준 경로
LOCAL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

# 도커에서 돌면 /app/config 가 존재하고, 로컬에서 돌면 ./config 를 쓴다.
CONFIG_DIR = DOCKER_CONFIG_DIR if DOCKER_CONFIG_DIR.exists() else LOCAL_CONFIG_DIR


def load_yaml(filename: str) -> Dict[str, Any]:
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_base_config() -> Dict[str, Any]:
    return load_yaml("base_config.yml")


def load_ingestion_file_config() -> Dict[str, Any]:
    return load_yaml("ingestion_file.yml")


def load_spark_jobs_config() -> Dict[str, Any]:
    return load_yaml("spark_jobs.yml")