import os
import json
import base64
from pathlib import Path
from typing import Any

import joblib
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError

from cnnClassifier import logger


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read a YAML file and return its contents as a ConfigBox."""
    try:
        with open(path_to_yaml, encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)

            if content is None:
                raise ValueError("YAML file is empty")

            logger.info(
                f"YAML file: {path_to_yaml} loaded successfully"
            )

            return ConfigBox(content)

    except BoxValueError:
        raise ValueError("YAML file is empty")


def create_directories(
    path_to_directories: list[Path],
    verbose: bool = True
) -> None:
    """Create directories if they do not already exist."""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)

        if verbose:
            logger.info(f"Created directory at: {path}")


def save_json(path: Path, data: dict) -> None:
    """Save dictionary data as a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    logger.info(f"JSON file saved at: {path}")


def load_json(path: Path) -> ConfigBox:
    """Load a JSON file and return it as a ConfigBox."""
    with open(path, encoding="utf-8") as f:
        content = json.load(f)

    logger.info(f"JSON file loaded successfully from: {path}")

    return ConfigBox(content)


def save_bin(data: Any, path: Path) -> None:
    """Save data as a binary file using joblib."""
    joblib.dump(value=data, filename=path)

    logger.info(f"Binary file saved at: {path}")


def load_bin(path: Path) -> Any:
    """Load a binary file using joblib."""
    data = joblib.load(path)

    logger.info(f"Binary file loaded from: {path}")

    return data


def get_size(path: Path) -> str:
    """Return file size in KB."""
    size_in_kb = round(os.path.getsize(path) / 1024)

    return f"~ {size_in_kb} KB"


def decode_image(imgstring: str, file_name: str) -> None:
    """Decode a base64 image string and save it to a file."""
    imgdata = base64.b64decode(imgstring)

    with open(file_name, "wb") as f:
        f.write(imgdata)


def encode_image_into_base64(cropped_image_path: str) -> bytes:
    """Encode an image file into a base64 byte string."""
    with open(cropped_image_path, "rb") as f:
        return base64.b64encode(f.read())