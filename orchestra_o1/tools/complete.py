"""CompleteTool: mark task as complete with final answer (GAIA only)."""
from __future__ import annotations

from typing import Any, Dict

from pydantic import Field

from base.agent.base_action import BaseAction
from base.engine.logs import logger


class CompleteTool(BaseAction):
    """Mark task as complete with final answer (GAIA only)."""
    
    name: str = "complete"
    description: str = "Mark the task as complete and provide the final answer"
    parameters: Dict[str, Any] = Field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "The final answer to the question"},
        },
        "required": ["answer"]
    })
    
    async def __call__(self, answer: str = "") -> Dict:
        """Execute complete action."""
        logger.info(f"[CompleteTool] Task completed with answer: {answer}")
        return {
            "success": True,
            "answer": answer,
            "done": True,
        }
