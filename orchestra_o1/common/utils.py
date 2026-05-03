"""Common utility functions"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def parse_json_response(resp: str) -> Dict[str, Any]:
    """Parse JSON response that may contain markdown code blocks
    
    Args:
        resp: LLM response string, possibly wrapped in ```json
        
    Returns:
        Parsed dict
    """
    s = resp.strip()
    if "```" in s:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
        if match:
            s = match.group(1)
    else:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start : end + 1]
    return json.loads(s)


def indent_text(text: str, indent: str = "   ") -> str:
    """Add indentation to each line of text
    
    Args:
        text: Original text
        indent: Indentation string
        
    Returns:
        Indented text
    """
    return "\n".join(indent + line for line in text.strip().split("\n"))


def format_tools_description(tools: List[Any], verbose: bool = False) -> str:
    """Generate tool description text for prompt usage
    
    Args:
        tools: List of tools, each tool needs name, description, parameters attributes
        verbose: Whether to use verbose format
        
    Returns:
        Formatted tool description string
    """
    if not tools:
        return "No tools available."
    
    if verbose:
        descriptions = []
        for tool in tools:
            desc = f"""Tool Name: {tool.name}
Description: {tool.description}
Parameters: {json.dumps(tool.parameters, indent=2)}"""
            descriptions.append(desc)
        return "\n\n".join(descriptions)
    else:
        return "\n\n".join([
            f"{t.name}: {t.description}\nParams: {json.dumps(t.parameters, indent=2)}"
            for t in tools
        ])
