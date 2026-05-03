from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from pydantic import BaseModel, Field

class BasicInfo(BaseModel):
    env_id: str
    instruction: str
    action_space: str
    max_steps: int
    meta_data: Dict[str, Any] = Field(default_factory=dict)

Observation = Dict[str, Any]
Action = Dict[str, Any]

class Environment(ABC):
    @abstractmethod
    def get_basic_info(self) -> BasicInfo:
        raise NotImplementedError

    @abstractmethod
    def reset(self, seed: int | None = None) -> Observation:
        """
        Reset the environment, return initial observation.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Accept an action and return (observation, reward, done, info).
        """
        raise NotImplementedError
