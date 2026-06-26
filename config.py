import yaml
from pathlib import Path

class ConfigLoader:
    @staticmethod
    def from_yaml(filepath: str) -> dict:
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return data or {}