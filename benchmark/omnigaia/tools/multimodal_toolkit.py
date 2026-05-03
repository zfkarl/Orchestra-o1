from __future__ import annotations

from typing import Any, Dict, Optional
import base64
import os
from pathlib import Path
from openai import AsyncOpenAI 
from base.agent.base_action import BaseAction
from base.engine.async_llm import LLMsConfig, llm_retry


## Image Tools
class ImageAnalysisAction(BaseAction):
    name: str = "ImageAnalysisAction"
    description: str = (
        "Call the multimodal model to conduct image analysis via given queries. "
        "Uses model_config.yaml for API configuration."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Instruction or question describing what to analyze in the image",
            },
            "image_path": {
                "type": "string",
                "description": "Local file path or URL that points to the image resource",
            }
        },
        "required": ["query", "image_path"],
        "additionalProperties": False,
    }
    
    # Default model for image analysis (should support vision)
    DEFAULT_MODEL: str = "gpt-5"

    def encode_image(self, image_path: str) -> str:
        data = Path(image_path).expanduser().read_bytes()
        return base64.b64encode(data).decode("utf-8")
    
    def _get_llm_config(self, model_name: Optional[str] = None) -> tuple:
        """Get LLM configuration from model_config.yaml or environment variables."""
        model_name = model_name or self.DEFAULT_MODEL
        
        # Try to get config from LLMsConfig (model_config.yaml)
        try:
            llms_config = LLMsConfig.default()
            model_config = llms_config.get(model_name)
            if model_config:
                return (
                    model_config.key,
                    model_config.base_url,
                    model_config.model,
                )
        except Exception:
            pass
        
        # Fallback to environment variables
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_API_BASE") or os.getenv("OPENAI_BASE_URL") or "https://newapi.deepwisdom.ai/v1"
        
        return (api_key, base_url, model_name)

    async def __call__(self, **kwargs) -> Any:
        query = kwargs.get("query")
        image_path = kwargs.get("image_path")

        if not query or not image_path:
            return {"success": False, "output": None, "error": "Both query and image_path are required.", "metrics": {}}

        api_key, base_url, model = self._get_llm_config()
        if not api_key:
            return {"success": False, "output": None, "error": "No API key found. Configure in model_config.yaml or set OPENAI_API_KEY", "metrics": {}}

        try:
            from openai import AsyncOpenAI  # type: ignore
        except Exception:
            return {"success": False, "output": None, "error": "openai package not available", "metrics": {}}

        try:
            if image_path.startswith(("http://", "https://")):
                image_url = image_path
            else:
                encoded = self.encode_image(image_path)
                image_url = f"data:image/png;base64,{encoded}"
        except Exception as exc:
            return {"success": False, "output": None, "error": f"Failed to prepare image: {exc}", "metrics": {}}

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": str(query)},
                ],
            }
        ]

        try:
            completion = await llm_retry(
                client.chat.completions.create,
                model=model,
                messages=messages,
                label=f"ImageAnalysis({model})",
            )
        except Exception as exc:
            return {"success": False, "output": None, "error": f"Image analysis failed: {exc}", "metrics": {}}

        content = None
        try:
            if completion and completion.choices:
                content = completion.choices[0].message.content
        except Exception:
            content = None

        if not content:
            return {"success": False, "output": None, "error": "Model returned empty response", "metrics": {}}

        return {"success": True, "output": content.strip(), "error": None, "metrics": {}}
