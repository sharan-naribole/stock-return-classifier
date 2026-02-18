"""
Config loader utility — loads and validates JSON configs with dot-notation access.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path


class ConfigLoader:
    """Loads and validates data_config.json and model_config.json."""

    def __init__(self, data_config_path: str, model_config_path: str):
        self.data_config_path = Path(data_config_path)
        self.model_config_path = Path(model_config_path)
        self._data_config = self._load_json(self.data_config_path)
        self._model_config = self._load_json(self.model_config_path)
        self._resolve_auto_dates()
        self._validate()

    def _load_json(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            return json.load(f)

    def _resolve_auto_dates(self):
        """Resolve 'auto' end_date to yesterday's date (last complete trading day)."""
        if self._data_config.get("end_date") == "auto":
            yesterday = datetime.now() - timedelta(days=1)
            self._data_config["end_date"] = yesterday.strftime("%Y-%m-%d")

    def _validate(self):
        """Basic validation of required config fields."""
        required_data = ["ticker", "start_date", "end_date"]
        for field in required_data:
            if field not in self._data_config:
                raise ValueError(f"Missing required data_config field: {field}")

        required_model = ["project_name", "eval_metric", "models"]
        for field in required_model:
            if field not in self._model_config:
                raise ValueError(f"Missing required model_config field: {field}")

    def get(self, key: str, default=None):
        """
        Dot-notation accessor for both configs.
        Searches data_config first, then model_config.
        Example: config.get("split.n_folds")
        """
        parts = key.split(".")
        for config in [self._data_config, self._model_config]:
            val = self._nested_get(config, parts)
            if val is not None:
                return val
        return default

    def _nested_get(self, d: dict, parts: list):
        """Recursively get nested dict value by key parts."""
        if not parts or not isinstance(d, dict):
            return None
        if len(parts) == 1:
            return d.get(parts[0])
        return self._nested_get(d.get(parts[0], {}), parts[1:])

    @property
    def data_config(self) -> dict:
        return self._data_config

    @property
    def model_config(self) -> dict:
        return self._model_config

    def get_output_dir(self, dir_type: str, project_name: str = None) -> Path:
        """Get absolute output directory path, creating it if needed."""
        if project_name is None:
            project_name = self._model_config.get("project_name", "default")
        base = self._data_config.get("output", {}).get(dir_type, dir_type)
        path = Path(base) / project_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def __repr__(self):
        return (
            f"ConfigLoader(ticker={self._data_config.get('ticker')}, "
            f"project={self._model_config.get('project_name')})"
        )
