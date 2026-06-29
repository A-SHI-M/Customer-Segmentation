from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    random_seed: int
    data_raw: Path
    data_processed: Path
    models_dir: Path
    numerical_features: List[str]
    engineered_features: List[str]
