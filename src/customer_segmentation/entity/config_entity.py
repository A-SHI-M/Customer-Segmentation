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


@dataclass(frozen=True)
class ModelEvaluationConfig:
    model_path: Path
    scaler_path: Path
    data_processed: Path
    artifacts_dir: Path
    metrics_path: Path
    experiment_name: str


@dataclass(frozen=True)
class ModelTrainerConfig:
    data_processed: Path
    scaler_path: Path
    model_path: Path
    artifacts_dir: Path
    cluster_profiles_path: Path
    experiment_name: str
    feature_cols: List[str]
    k_min: int
    k_max: int
    n_init: int
    max_iter: int
    random_seed: int
