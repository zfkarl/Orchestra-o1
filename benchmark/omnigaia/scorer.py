"""OmniGAIA scoring utilities.

Provides utility functions for prediction extraction, path resolution, and file preprocessing.
LLM-based scoring is handled by llm_scorer.py.
"""

from __future__ import annotations

import os
from typing import Any, Tuple


def extract_pred_text(prediction: Any) -> str:
    """
    Extract prediction text from various formats.
    
    Handles:
    - None -> ""
    - dict with keys: final_answer, output, result, text
    - str -> str
    - other -> str(other)
    
    Args:
        prediction: The model's prediction in various formats
        
    Returns:
        Extracted text string
    """
    if prediction is None:
        return ""
    if isinstance(prediction, dict):
        # Try common keys in order of preference
        for key in ("final_answer", "answer", "output", "result", "text"):
            if key in prediction and prediction.get(key) is not None:
                return str(prediction.get(key))
        # If has success=False, return empty
        if "success" in prediction and not prediction.get("success"):
            return ""
        return str(prediction)
    return str(prediction)


def calculate_score(expected_output: Any, prediction: Any) -> Tuple[float, str]:
    """
    Calculate score with prediction extraction.
    Uses LLM-based scoring via llm_scorer.

    Args:
        expected_output: The expected correct answer
        prediction: The model's prediction (can be dict, str, or None)
        
    Returns:
        Tuple of (score, extracted_text)
    """
    from benchmark.omnigaia.llm_scorer import llm_semantic_score_sync

    pred_text = extract_pred_text(prediction)
    if expected_output is None:
        return (1.0 if pred_text else 0.0, pred_text)
    score = llm_semantic_score_sync(pred_text, str(expected_output))
    return (score, pred_text)


def resolve_gaia_attachment_path(file_name: str, file_root: str | None = None) -> str:
    """
    Resolve GAIA attachment file path.
    
    Args:
        file_name: The attachment file name
        file_root: Optional root directory for attachments
        
    Returns:
        Resolved file path
    """
    if not file_name:
        return ""

    # If it's already an absolute path, keep it
    if os.path.isabs(file_name):
        return file_name

    # If caller supplied a root, use it
    if file_root:
        root = str(file_root).rstrip("/\\")
        return os.path.join(root, file_name)

    # Default GAIA validation attachment root
    default_root = os.path.join("benchmark", "gaia", "data", "Gaia", "2023", "validation")
    return os.path.join(default_root, file_name)


def preprocess_file(task: str, file_name: str | None, file_root: str | None = None) -> str:
    """
    Preprocess task description with file attachment hint.
    
    Args:
        task: The task description
        file_name: Optional attachment file name
        file_root: Optional root directory for attachments
        
    Returns:
        Task description with file hint appended if file_name is provided
    """
    if not file_name:
        return task
    hint_path = resolve_gaia_attachment_path(file_name, file_root=file_root)
    return (
        f"{task}\n"
        f"(* LOCAL FILES attached: {hint_path} - "
        f"Use ImageAnalysisAction for images, ParseAudioAction for audio, "
        f"ExecuteCodeAction for other files like txt/csv/xlsx/json/pdb)"
    )
