'''
The code of this tool is written with reference to https://github.com/hkust-nlp/WebExplorer/blob/master/src/inference/tool_webexplorer_browse.py.
'''

from __future__ import annotations

from typing import Any, Dict, List, Optional
import asyncio
import os
import requests
import tiktoken
from openai import AsyncOpenAI 
from base.agent.base_action import BaseAction
from base.engine.async_llm import LLMsConfig, llm_retry


class ExtractUrlContentAction(BaseAction):
    name: str = "ExtractUrlContentAction"
    description: str = "Fetch and extract text content from a web URL (http/https only). NOTE: This tool can ONLY read online web pages, NOT local files or images. Requires JINA_API_KEY."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "browse_query": {"type": "string"},
        },
        "required": ["url", "browse_query"],
        "additionalProperties": False,
    }
    
    # Default model for content extraction
    DEFAULT_MODEL: str = "deepseek-reasoner"

    def read_jina_page(self, url, JINA_API_KEY, max_retry=5):
        for attempt in range(max_retry):
            headers = {
                "Authorization": f"Bearer {JINA_API_KEY}",
            }

            try:
                response = requests.get(
                    f"https://r.jinaai.cn/{url}",
                    headers=headers,
                    timeout=50
                )
                if response.status_code == 200:
                    webpage_content = response.text
                    return webpage_content
                else:
                    print(f"Jina API error: {response.text}")
                    raise ValueError("jina readpage error")
            except Exception as e:
                print(f"jina_readpage {attempt} error: {e}", flush=True)
                if attempt == max_retry - 1:
                    return "[Jina] Failed to read page."
    
    def _get_llm_config(self, model_name: Optional[str] = None) -> tuple:
        """
        Get LLM configuration from model config or environment variables.
        
        Returns:
            tuple: (api_key, base_url, model)
        """
        model_name = model_name or self.DEFAULT_MODEL
        
        # Try to get config from LLMsConfig
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
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_API_BASE") or "https://api.openai.com/v1"
        
        return (api_key, base_url, model_name)
    
    async def call_openai_model(self, query: str) -> str:
        api_key, base_url, model = self._get_llm_config()
        
        if not api_key:
            raise RuntimeError("No API key found. Set OPENAI_API_KEY or configure in model_config.yaml")
        
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": str(query)},
                ],
            }
        ]

        try:
            completion = await llm_retry(
                client.chat.completions.create,
                model=model,
                messages=messages,
                label=f"ExtractUrlContent({model})",
            )
        except Exception as exc:
            raise RuntimeError(f"[JINA] Call OpenAI model failed: {exc}") from exc

        try:
            message = completion.choices[0].message.content if completion and completion.choices else None
        except Exception as exc:
            raise RuntimeError(f"[JINA] Unexpected response format: {exc}") from exc

        if not message:
            raise RuntimeError("[JINA] Model returned empty response")

        return message.strip()

    async def __call__(self, **kwargs) -> Any:
        JINA_API_KEY = os.getenv("JINA_API_KEY")
        if not JINA_API_KEY :
            return {"success": False, "output": None, "error": "JINA_API_KEY is not set.", "metrics": {}}
        url = kwargs.get("url")
        browse_query = kwargs.get("browse_query")
        
        source_text = self.read_jina_page(url, JINA_API_KEY=JINA_API_KEY)
        print(source_text)
        
        if not isinstance(source_text, str):
            return {"success": False, "output": None, "error": "Failed to extract content from the targeted url.", "metrics": {}}

        if source_text.strip() == '' or source_text.startswith("[Jina] Failed to read page"):
            return {"success": False, "output": None, "error": "Failed to extract content from the targeted url.", "metrics": {}}
    
        query = f"Please read the source content and answer a following question:\n---begin of source content---\n{source_text}\n---end of source content---\n\nIf there is no relevant information, please clearly refuse to answer. Now answer the question based on the above content:\n{browse_query}"
    
        # Handle long content chunking (following deep_research_utils.py logic)
        encoding = tiktoken.get_encoding("cl100k_base")
        tokenized_source_text = encoding.encode(source_text)

        if len(tokenized_source_text) > 95000:  # Using same token limit as original code
            output = "Since the content is too long, the result is split and answered separately. Please combine the results to get the complete answer.\n"
            num_split = max(2, len(tokenized_source_text) // 95000 + 1)
            chunk_len = len(tokenized_source_text) // num_split
            print(f"Browse too long with length {len(tokenized_source_text)}, split into {num_split} parts, with each part length {chunk_len}", flush=True)
            
            tasks: List[asyncio.Task[str]] = []
            for i in range(num_split):
                start_idx = i * chunk_len
                end_idx = min(start_idx + chunk_len + 1024, len(tokenized_source_text))
                source_text_i = encoding.decode(tokenized_source_text[start_idx:end_idx])
                query_i = f"Please read the source content and answer a following question:\n--- begin of source content ---\n{source_text_i}\n--- end of source content ---\n\nIf there is no relevant information, please clearly refuse to answer. Now answer the question based on the above content:\n{browse_query}"

                tasks.append(asyncio.create_task(self.call_openai_model(query_i)))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    return {"success": False, "output": None, "error": str(result), "metrics": {}}
                output += f"--- begin of result part {idx+1} ---\n{result}\n--- end of result part {idx+1} ---\n\n"
        else:
            try:
                output = await self.call_openai_model(query)
            except Exception as exc:
                return {"success": False, "output": None, "error": str(exc), "metrics": {}}
        
        if output is None or output.strip() == "":
            return {"success": False, "output": None, "error": "[JINA] Browse error with empty output.", "metrics": {}}

        # Wrap result with URL and a short preview for logging/traceability
        preview = source_text[:2000] if isinstance(source_text, str) else ""
        packaged = {"url": url, "answer": output, "source_preview": preview}
        return {"success": True, "output": packaged, "error": None, "metrics": {}}
