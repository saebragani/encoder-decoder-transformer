import yaml

class ConfigLoader:
    @staticmethod
    def from_yaml(filepath: str) -> dict:
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return data or {}