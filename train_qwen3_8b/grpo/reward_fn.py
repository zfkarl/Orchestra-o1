#!/usr/bin/env python3
"""
Orchestra-o1 Main-Agent GRPO reward (v3 - unified LLM-as-judge multi-dimensional scoring).

verl entry point: reward.custom_reward_function.name=compute_score
Signature: compute_score(data_source, solution_str, ground_truth, extra_info)

total_score = w_format   * r_format          (0.10)  <- LLM judge 0/1 binary
     + w_action   * r_action_valid     (0.10)  <- LLM judge 0/1 binary
     + w_tool     * r_tool_reasonable  (0.20)  <- LLM judge 0-3 4-level
     + w_decision * r_decision_quality (0.60 * core dimension) <- LLM judge 0-3 4-level

All dimensions evaluated by LLM judge (claude-haiku-4-5) scored simultaneously in one call:
  - format_correct and action_valid use 0/1 binary scoring
  - tool_reasonable and decision_quality use 0-3 4-level scoring, normalized to [0,1]

**Scoring dimension description**:
  - format_correct: LLM judge 0/1 score, whether JSON format is correct and fields are complete
  - action_valid: LLM judge 0/1 score, whether action is valid and parameter structure is correct
  - tool_reasonable: LLM judge 0-3 score, whether tool selection and subtask assignment are reasonable
  - decision_quality: LLM judge 0-3 score, overall decision quality, references GPT-5 expert trajectory but encourages exploration
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import traceback
from typing import Any

import requests

# ====== weight ======
W_FORMAT = 0.10        # format correctness
W_ACTION = 0.10        # action validity
W_TOOL = 0.20          # tool call reasonableness
W_DECISION = 0.60      # decision quality(core dimension)

VALID_ACTIONS = {"delegate_task", "complete"}

# ====== LLM Judge  ======
LLM_JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL", "claude-haiku-4-5")
LLM_JUDGE_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/")
LLM_JUDGE_API_KEY = os.environ.get("OPENAI_API_KEY", "your_api_key_here")
LLM_JUDGE_TIMEOUT = 60  # seconds
LLM_JUDGE_MAX_RETRIES = 5  # Max retries (6 total attempts)
LLM_JUDGE_BASE_BACKOFF = 2.0  # Exponential backoff base (seconds)
LLM_JUDGE_MAX_BACKOFF = 60.0  # Max backoff time (seconds)
LLM_JUDGE_JITTER = 1.0  # Random jitter range (seconds)

# -------------------- JSON  --------------------
_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def _try_parse_json(text: str) -> dict | None:
    """Try to extract a JSON object from model output."""
    if not text:
        return None

    # 1) ```json ... ```
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 2)  {...}
    m = _BRACE_RE.search(text)
    if m:
        snippet = m.group(0)
        for end in range(len(snippet), 0, -1):
            if snippet[end - 1] != "}":
                continue
            try:
                return json.loads(snippet[:end])
            except json.JSONDecodeError:
                continue

    # 3) 
    try:
        return json.loads(text)
    except Exception:
        return None


# -------------------- LLM Judge --------------------
def _call_llm_judge(prompt: str) -> str | None:
    """call claude-haiku-4-5 as judge, return model reply text.return None on failure."""
    url = LLM_JUDGE_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_JUDGE_API_KEY}",
    }
    payload = {
        "model": LLM_JUDGE_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }

    for attempt in range(LLM_JUDGE_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                url, headers=headers, json=payload,
                timeout=LLM_JUDGE_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < LLM_JUDGE_MAX_RETRIES:
                backoff = min(
                    LLM_JUDGE_BASE_BACKOFF * (2 ** attempt),
                    LLM_JUDGE_MAX_BACKOFF,
                )
                jitter = random.uniform(0, LLM_JUDGE_JITTER)
                sleep_time = backoff + jitter
                print(f"[reward_fn] LLM judge attempt {attempt + 1} failed ({e}), "
                      f"retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                continue
            print(f"[reward_fn] LLM judge call failed after {LLM_JUDGE_MAX_RETRIES + 1} attempts: {e}")
            return None


# --------------------  judge  --------------------
_SCORE_PATTERN = re.compile(
    r"FORMAT_CORRECT\s*[:=]\s*(\d)\s*.*?"
    r"ACTION_VALID\s*[:=]\s*(\d)\s*.*?"
    r"TOOL_REASONABLE\s*[:=]\s*(\d)\s*.*?"
    r"DECISION_QUALITY\s*[:=]\s*(\d)",
    re.DOTALL | re.IGNORECASE,
)


def _parse_judge_scores(response: str | None) -> dict[str, float]:
    """Parse from judge reply 4 dimension scores, normalized to [0,1].

    - FORMAT_CORRECT and ACTION_VALID: 0/1 binary, used directly as float
    - TOOL_REASONABLE and DECISION_QUALITY: 0-3 4-level, divided by 3 to normalize

    Expected judge reply format:
        FORMAT_CORRECT: 1
        ACTION_VALID: 1
        TOOL_REASONABLE: 3
        DECISION_QUALITY: 2

    Parse strategy:
    1. Prefer searching from end of reply, because judge may mention these keywords during analysis
    2. Only match "KEY: <digit>" format lines(at line start or with minimal prefix)
    3. Fall back to regex full-text match

    Return all zeros on parse failure.
    """
    default = {
        "format_correct": 0.0,
        "action_valid": 0.0,
        "tool_reasonable": 0.0,
        "decision_quality": 0.0,
    }
    if not response:
        return default

    key_map = {
        "FORMAT_CORRECT": "format_correct",
        "ACTION_VALID": "action_valid",
        "TOOL_REASONABLE": "tool_reasonable",
        "DECISION_QUALITY": "decision_quality",
    }

    # 0/1 binary(do not do /3 )
    binary_keys = {"format_correct", "action_valid"}

    # Strategy 1:Search from end line by line for strict "KEY: <digit>" 
    _STRICT_LINE_RE = re.compile(
        r"^[\s\-\*#>]*"
        r"(FORMAT_CORRECT|ACTION_VALID|TOOL_REASONABLE|DECISION_QUALITY)"
        r"\s*[:=]\s*(\d)",
        re.IGNORECASE,
    )
    scores = {}
    lines = response.split("\n")
    # , 
    for line in reversed(lines):
        m = _STRICT_LINE_RE.match(line.strip())
        if m:
            raw_key = m.group(1).upper()
            if raw_key in key_map:
                norm_key = key_map[raw_key]
                if norm_key not in scores:  # only take the last occurrence
                    val = int(m.group(2))
                    if norm_key in binary_keys:
                        scores[norm_key] = min(val, 1) * 1.0  # 0/1 binary
                    else:
                        scores[norm_key] = min(val, 3) / 3.0  # 0-3 normalized to [0,1]

    if len(scores) == 4:
        return scores

    # Strategy 2:regex full-text match(consecutive 4 key-values)
    m = _SCORE_PATTERN.search(response)
    if m:
        try:
            return {
                "format_correct": min(int(m.group(1)), 1) * 1.0,
                "action_valid": min(int(m.group(2)), 1) * 1.0,
                "tool_reasonable": min(int(m.group(3)), 3) / 3.0,
                "decision_quality": min(int(m.group(4)), 3) / 3.0,
            }
        except (ValueError, IndexError):
            pass

    # Fill missing with 0
    for k in default:
        if k not in scores:
            scores[k] = 0.0
    return scores


# -------------------- Unified Judge Prompt --------------------

_JUDGE_PROMPT = """You are an expert judge evaluating an AI agent's output in a multi-step task-solving pipeline.

The agent (Main Agent) orchestrates sub-agents to solve complex tasks. At each step, it outputs a JSON decision that either:
- **delegate_task**: Break the problem into sub-tasks and assign them to sub-agents (each sub-task should have task_instruction, model, and optionally tools)
- **complete**: Provide the final answer (should have params.answer)

You will evaluate the agent's output on 4 dimensions. FORMAT_CORRECT and ACTION_VALID are scored 0 or 1 (binary). TOOL_REASONABLE and DECISION_QUALITY are scored 0-3 (integer only).

## Original Question
{question}

## Ground Truth Answer
{ground_truth}

## Current Step Context (Subtask History)
{subtask_history}

## GPT-5 Expert's Decision (reference, NOT the only valid approach)
- Action: {expert_action}
- Expert Output:
```json
{expert_json}
```

## Agent's Raw Output (to be evaluated)
```
{pred_raw}
```

## Agent's Parsed Decision
```json
{pred_json}
```

## Scoring Dimensions

### 1. FORMAT_CORRECT (0 or 1)
Is the agent's output a valid JSON decision with required fields?
- 1: Valid JSON with "action" field present and correctly structured
- 0: Not valid JSON, or missing "action" field, or completely unparseable

### 2. ACTION_VALID (0 or 1)
Is the chosen action valid and properly parameterized?
- 1: Action is valid ("delegate_task" or "complete") with "params" field present
- 0: Action is not in the valid set, or "params" field is missing/invalid

### 3. TOOL_REASONABLE (0-3)
Are the tool choices and sub-task assignments reasonable? (For "complete" action, evaluate whether completing at this point is appropriate)
- 3: Excellent tool/model selection, sub-tasks are well-scoped and clearly instructed
- 2: Acceptable tool selection but could be improved (e.g., missing a useful tool, overly broad instructions)
- 1: Questionable or mostly inappropriate tool choices, poorly defined sub-tasks
- 0: No tools specified when needed, or completely irrelevant assignments

### 4. DECISION_QUALITY (0-3) * Most Important
Overall decision quality — does this decision make good progress toward solving the problem?

**Key principle: We encourage exploration. The agent does NOT need to copy the expert's exact strategy.**

- 3: Excellent decision — closely aligned with expert's approach, OR takes a different but equally valid/creative approach, OR directly provides the correct answer
- 2: Acceptable decision — reasonable strategy but with notable inefficiencies or differences from optimal
- 1: Poor decision — partially relevant but unlikely to lead to the correct answer, or fundamentally flawed
- 0: Completely wrong — irrelevant output, nonsensical, or harmful to solving the task

**When scoring DECISION_QUALITY, consider:**
- If the agent's approach differs from the expert but is still reasonable and could lead to the correct answer → score 2-3
- If the agent chose "complete" and the answer matches the ground truth → score 3 regardless of expert action
- If the agent chose "complete" but the answer is wrong when expert says delegate → score 0
- If the agent chose "delegate_task" with reasonable sub-tasks when expert says complete → score 1-2 (inefficient but not wrong)

## Your Task
Evaluate the agent's output and provide scores for each dimension.

**IMPORTANT: Output ONLY the 4 scores below. Do NOT include any explanation, analysis, or reasoning. Just the scores.**

FORMAT_CORRECT: <score>
ACTION_VALID: <score>
TOOL_REASONABLE: <score>
DECISION_QUALITY: <score>"""


# --------------------  judge prompt call --------------------
def _evaluate_with_llm_judge(
    solution_str: str,
    decision: dict | None,
    extra_info: dict,
) -> dict[str, float]:
    """
    Use LLM judge for multi-dimensional scoring of model output.

    Returns dict: {format_correct, action_valid, tool_reasonable, decision_quality}
    - format_correct / action_valid: 0.0 or 1.0 (binary)
    - tool_reasonable / decision_quality: [0, 1] range (0-3 normalized)
    """
    question = str(extra_info.get("question", "N/A"))
    ground_truth = str(extra_info.get("answer", "N/A"))
    subtask_history = str(extra_info.get("subtask_history", "No subtasks completed yet."))

    # Get expert info
    has_expert = bool(extra_info.get("has_expert"))
    if has_expert:
        expert_action = str(extra_info.get("expert_action", "unknown"))
        expert_params = extra_info.get("expert_params") or {}
        expert_json_str = json.dumps(
            {"action": expert_action, "params": expert_params},
            ensure_ascii=False, indent=2,
        )
    else:
        expert_action = "unknown"
        expert_json_str = '{"note": "No expert decision available"}'

    # Get pred info
    pred_json_str = json.dumps(decision, ensure_ascii=False, indent=2) if decision else "null (JSON parse failed)"

    # Truncate long raw output to avoid prompt overflow
    pred_raw = solution_str[:3000] if len(solution_str) > 3000 else solution_str

    prompt = _JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        subtask_history=subtask_history,
        expert_action=expert_action,
        expert_json=expert_json_str,
        pred_raw=pred_raw,
        pred_json=pred_json_str,
    )

    response = _call_llm_judge(prompt)
    scores = _parse_judge_scores(response)

    return scores


# -------------------- verl entry --------------------
def _compute_score_inner(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> dict:
    extra_info = extra_info or {}

    decision = _try_parse_json(solution_str)

    # call LLM judge 
    scores = _evaluate_with_llm_judge(solution_str, decision, extra_info)

    r_format = scores["format_correct"]    # 0/1 binary
    r_action = scores["action_valid"]      # 0/1 binary
    r_tool = scores["tool_reasonable"]     # 0-3 normalized to [0,1]
    r_decision = scores["decision_quality"]  # 0-3 normalized to [0,1]

    # Weighted sum
    total_score = (
        W_FORMAT * r_format
        + W_ACTION * r_action
        + W_TOOL * r_tool
        + W_DECISION * r_decision
    )
    total_score = max(0.0, min(1.0, total_score))

    # acc: decision_quality >= 0.67 (judge score 2 or 3) and format >= 0.67 => acc=1
    acc = 1.0 if (r_decision >= 0.67 and r_format >= 0.67) else 0.0

    return {
        "score": total_score,
        "acc": acc,
        "format_correct": r_format,
        "action_valid": r_action,
        "tool_reasonable": r_tool,
        "decision_quality": r_decision,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> dict:
    """verl callback: score a single rollout. Outer try/except as safety net to avoid crashing the entire batch."""
    try:
        return _compute_score_inner(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )
    except Exception as e:
        print(f"[reward_fn] ERROR in compute_score: {e}")
        traceback.print_exc()
        return {
            "score": 0.0,
            "acc": 0.0,
            "format_correct": 0.0,
            "action_valid": 0.0,
            "tool_reasonable": 0.0,
            "decision_quality": 0.0,
        }


# -------------------- Quick self-test --------------------
if __name__ == "__main__":
    extra = {
        "has_expert": True,
        "step_index": 1,
        "question": "What is the drain current?",
        "expert_action": "delegate_task",
        "expert_params": {
            "tasks": [
                {
                    "task_instruction": "Compute k from the reference and calculate I_D.",
                    "model": "model_1",
                    "tools": ["ExecuteCodeAction"],
                }
            ]
        },
        "expert_tools": ["ExecuteCodeAction"],
        "expert_task_instructions": ["Compute k from the reference and calculate I_D."],
        "expert_num_subtasks": 1,
        "expert_answer": "",
        "answer": "106",
        "subtask_history": "No subtasks completed yet.",
    }

    sample_good = """```json
{
  "action": "delegate_task",
  "reasoning": "need to compute drain current using the given formula",
  "params": {
    "tasks": [
      {"task_instruction": "Compute k from reference and calculate I_D using the saturation equation.",
       "model": "model_1", "tools": ["ExecuteCodeAction"]}
    ]
  }
}
```"""
    sample_complete_wrong = """{"action": "complete", "params": {"answer": "123"}}"""
    sample_complete_right = """{"action": "complete", "params": {"answer": "106"}}"""
    sample_bad_format = """This is not JSON at all, just random text."""

    print("=== Test complete compute_score (requires network access to claude-haiku-4-5) ===")
    print("\n1. GOOD delegate  ->")
    result = compute_score("x", sample_good, "106", extra)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n2. complete WRONG ->")
    result = compute_score("x", sample_complete_wrong, "106", extra)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n3. complete RIGHT ->")
    result = compute_score("x", sample_complete_right, "106", extra)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n4. BAD format     ->")
    result = compute_score("x", sample_bad_format, "106", extra)
    print(json.dumps(result, indent=2, ensure_ascii=False))
