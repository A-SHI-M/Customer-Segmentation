from pathlib import Path
from customer_segmentation.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from customer_segmentation.utils import read_yaml, create_directories
from customer_segmentation.entity.config_entity import DataIngestionConfig


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config["data_ingestion"]
        create_directories([config["root_dir"]])
        return DataIngestionConfig(
            root_dir=Path(config["root_dir"]),
            source_url=config["source_url"],
            local_data_file=Path(config["local_data_file"]),
            unzip_dir=Path(config["unzip_dir"]),
        )
