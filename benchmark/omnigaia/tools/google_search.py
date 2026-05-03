from __future__ import annotations

from typing import Any, Dict, List
import os
import json

from base.agent.base_action import BaseAction

# Serper API configuration (official API)
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_BASE_URL = "https://google.serper.dev/search"


class GoogleSearchAction(BaseAction):
    name: str = "GoogleSearchAction"
    description: str = "Search via Serper (google.serper.dev) and return snippets. Requires SERPER_API_KEY."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer", "default": 5},
            "gl": {"type": "string", "default": "us"},
            "hl": {"type": "string", "default": "en"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    async def __call__(self, **kwargs) -> Any:
        # Directly use the new proxy API, don't read from env vars (avoid being overwritten by old key)
        api_key = SERPER_API_KEY
        base_url = SERPER_BASE_URL

        query = kwargs.get("query", "")
        k = int(kwargs.get("k", 5))
        gl = kwargs.get("gl", "us")
        hl = kwargs.get("hl", "en")

        try:
            import aiohttp  # type: ignore
        except Exception:
            return {"success": False, "output": None, "error": "aiohttp not available", "metrics": {}}

        async def _fetch() -> Dict[str, Any]:
            # Official Serper API uses X-API-KEY authentication
            headers = {
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            }
            payload = {"q": query, "num": k}
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(base_url, json=payload, headers=headers) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        raise RuntimeError(f"HTTP {resp.status}: {text[:500]}")
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError as e:
                        raise RuntimeError(f"JSON decode error: {e}, response: {text[:500]}")

        def _parse_results(results: Dict[str, Any]) -> List[Dict[str, str]]:
            snippets: List[Dict[str, str]] = []
            ans = results.get("answerBox") or {}
            if isinstance(ans, dict):
                if ans.get("answer"):
                    return [{"content": str(ans.get("answer")), "source": "None"}]
                if ans.get("snippet"):
                    return [{"content": str(ans.get("snippet")).replace("\n", " "), "source": "None"}]
                if ans.get("snippetHighlighted"):
                    return [{"content": str(ans.get("snippetHighlighted")), "source": "None"}]
            kg = results.get("knowledgeGraph") or {}
            if isinstance(kg, dict):
                title = kg.get("title")
                etype = kg.get("type")
                if etype:
                    snippets.append({"content": f"{title}: {etype}", "source": "None"})
                desc = kg.get("description")
                if desc:
                    snippets.append({"content": str(desc), "source": "None"})
                for attr, val in (kg.get("attributes") or {}).items():
                    snippets.append({"content": f"{attr}: {val}", "source": "None"})
            for item in (results.get("organic") or [])[: max(1, k)]:
                if "snippet" in item:
                    snippets.append({"content": str(item.get("snippet")), "source": str(item.get("link"))})
                for attr, val in (item.get("attributes") or {}).items():
                    snippets.append({"content": f"{attr}: {val}", "source": str(item.get("link"))})
            if not snippets:
                return [{"content": "No good Google Search Result was found", "source": "None"}]
            return snippets

        try:
            results = await _fetch()
            parsed = _parse_results(results)
            return {"success": True, "output": parsed, "error": None, "metrics": {}}
        except Exception as e:
            return {"success": False, "output": None, "error": str(e), "metrics": {}}

