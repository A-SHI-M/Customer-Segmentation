import os
import sys
import yaml
from pathlib import Path


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        _, _, exc_tb = error_detail.exc_info()
        self.error_message = (
            f"Error in [{exc_tb.tb_frame.f_code.co_filename}] "
            f"at line [{exc_tb.tb_lineno}]: {error_message}"
        )

    def __str__(self):
        return self.error_message


def read_yaml(path: Path) -> dict:
    with open(path) as f:
        content = yaml.safe_load(f)
    if content is None:
        raise ValueError(f"yaml file is empty: {path}")
    return content


def create_directories(paths: list) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)
