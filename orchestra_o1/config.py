"""Configuration for the Orchestra-o1 OmniGAIA benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class GAIAOrchestraConfig:
    """Orchestra-o1 configuration for the OmniGAIA benchmark."""

    # Model
    main_model: str
    sub_models: List[str]

    # Dataset
    dataset_path: Path
    attachments_dir: Path
    level_filter: List[int] | None = None
    max_tasks: int | None = None

    # Execution
    max_attempts: int = 5
    max_concurrency: int = 1

    # Output
    result_folder: Path = field(default_factory=lambda: Path("workspace/logs"))
    trajectory_folder: Path = field(default_factory=lambda: Path("workspace/logs/trajectories"))
    timestamp: str | None = None

    @classmethod
    def load(cls, config_path: Path | str) -> "GAIAOrchestraConfig":
        """Load configuration from a YAML file."""
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # Required fields
        main_model = raw.get("main_model")
        if not main_model:
            raise ValueError("main_model is required")

        sub_models = raw.get("sub_models")
        if not sub_models or not isinstance(sub_models, list):
            raise ValueError("sub_models must be a non-empty list")

        dataset_path = raw.get("dataset_path")
        if not dataset_path:
            raise ValueError("dataset_path is required")
        dataset_path = cls._resolve_path(dataset_path, config_path)

        attachments_dir = raw.get("attachments_dir")
        if not attachments_dir:
            raise ValueError("attachments_dir is required")
        attachments_dir = cls._resolve_path(attachments_dir, config_path)

        # Optional fields
        level_filter = raw.get("level_filter")
        max_tasks = raw.get("max_tasks")
        max_attempts = int(raw.get("max_attempts", 5))
        max_concurrency = int(raw.get("max_concurrency", 1))

        result_folder = cls._resolve_path(
            raw.get("result_folder", "workspace/logs"),
            config_path,
        )
        trajectory_folder = cls._resolve_path(
            raw.get("trajectory_folder", "workspace/logs/trajectories"),
            config_path,
        )

        return cls(
            main_model=str(main_model),
            sub_models=[str(m) for m in sub_models],
            dataset_path=dataset_path,
            attachments_dir=attachments_dir,
            level_filter=level_filter,
            max_tasks=max_tasks,
            max_attempts=max_attempts,
            max_concurrency=max_concurrency,
            result_folder=result_folder,
            trajectory_folder=trajectory_folder,
        )

    @staticmethod
    def _resolve_path(path_str: str, config_path: Path) -> Path:
        """Resolve a path (relative to config file or project root)."""
        path = Path(path_str)
        if path.is_absolute():
            return path
        # Try relative to config file
        rel_to_config = config_path.parent / path
        if rel_to_config.exists():
            return rel_to_config.resolve()
        # Try relative to project root
        PROJECT_ROOT = Path(__file__).parent.parent
        return (PROJECT_ROOT / path).resolve()
