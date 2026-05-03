"""
OmniGAIA-specific MainAgent prompt.

OmniGAIA tasks are multimodal question-answering tasks that require:
- Video analysis (frame extraction + vision model)
- Audio analysis (speech-to-text + understanding)
- Image analysis (vision model)
- Web search and information retrieval
- Code execution for computation
- Final answer extraction

This version supports parallel subtask assignment: MainAgent assigns multiple independent subtasks via delegate_task,
each executed by an independent SubAgent in parallel. For dependent subtasks, MainAgent can
execute in phases - complete prerequisite subtasks in parallel first, then plan follow-up subtasks based on results.
"""
import json
from typing import Any, Dict, List

from orchestra_o1.main_agent import build_model_pricing_table


def format_tools_description(tools: List[Any]) -> str:
    """Format tools list into description string."""
    if not tools:
        return "No tools available."

    descriptions = []
    for tool in tools:
        desc = f"Tool Name: {tool.name}\nDescription: {tool.description}"
        if tool.parameters:
            desc += f"\nParameters: {json.dumps(tool.parameters, indent=2)}"
        descriptions.append(desc)

    return "\n\n".join(descriptions)


class OmniGAIAMainAgentPrompt:
    """Generate prompts for OmniGAIA benchmark tasks (parallel subtask version)."""

    @staticmethod
    def build_prompt(
        instruction: str,
        meta: Dict[str, Any],
        prior_context: str,
        attempt_index: int,
        max_attempts: int,
        sub_models: List[str],
        subtask_history: str = "",
        model_to_alias: Dict[str, str] = None,
        tools: List[Any] = None,
    ) -> str:
        remaining_attempts = max_attempts - attempt_index
        model_pricing_table = build_model_pricing_table(sub_models, model_to_alias)
        tools_description = format_tools_description(tools or [])

        # Extract multimodal input info from metadata
        modal_inputs = meta.get("modal_inputs", [])
        modal_info = ""
        if modal_inputs:
            modal_lines = []
            for m in modal_inputs:
                modal_lines.append(f"  - Type: {m.get('type', 'unknown')}, ID: {m.get('id', '')}, Path: {m.get('path', 'N/A')}")
            modal_info = "\n".join(modal_lines)

        task_type = meta.get("task_type", "")
        category = meta.get("category", "")

        return f"""
You are the MainAgent (Orchestrator) for OmniGAIA benchmark tasks. Your role is to analyze the given QUESTION, plan a multi-phase execution strategy, and delegate subtasks to SubAgents — maximizing parallelism where possible while respecting task dependencies.

==== CORE PRINCIPLE: SMART PARALLEL DECOMPOSITION ====
Not all subtasks can run simultaneously. Some depend on others' results. Your job is to:
1. Identify which subtasks are INDEPENDENT and can run in parallel NOW
2. Identify which subtasks DEPEND on others' results and must wait for later phases
3. In each delegation round, submit ALL currently-runnable independent subtasks together
4. After receiving results, plan the NEXT round of subtasks based on what you learned

Key rules:
- Each subtask runs as an independent SubAgent with its own environment
- All subtasks within ONE delegation call execute simultaneously in parallel
- Always use the "tasks" list format (even for a single subtask)
- Each delegation (regardless of how many parallel subtasks) counts as ONE attempt

DECOMPOSITION STRATEGY:
Phase 1: Identify ALL sub-goals needed to answer the question
Phase 2: Classify each sub-goal:
  - INDEPENDENT: Can start immediately without any prior results (→ run in parallel NOW)
  - DEPENDENT: Needs results from other sub-goals first (→ plan for a LATER round)
Phase 3: Submit all INDEPENDENT sub-goals as parallel subtasks in this round
Phase 4: After receiving results, re-evaluate:
  - Are the results sufficient to answer the question? → Use 'complete'
  - Are there DEPENDENT sub-goals now unblocked? → Submit them as the next parallel batch
  - Do results reveal NEW sub-goals? → Add them to the plan

Examples of good phased parallel decomposition:

Example 1 — Video question with web search:
  Round 1 (parallel): Subtask A: Analyze video content, Subtask B: Search for general background info
  Round 2 (after results): Subtask C: Search for specific details mentioned in the video (depends on A's findings)

Example 2 — Multiple media files:
  Round 1 (parallel): One subtask per media file (all independent)
  Round 2 (if needed): Synthesize/verify findings with additional searches

Example 3 — Data extraction + computation:
  Round 1: Extract data from source (must come first)
  Round 2: Compute on extracted data (depends on Round 1 results)

Example 4 — Fully independent tasks:
  Round 1 (parallel): All subtasks at once (no dependencies, single round suffices)

MULTIMODAL TASK HANDLING:
This is an OmniGAIA task that may involve VIDEO, AUDIO, and IMAGE inputs.
- For VIDEO: Use VideoAnalysisAction to analyze both visual frames AND audio track. Set analyze_audio=true (default) for videos with speech/sound.
- For STANDALONE AUDIO FILES (mp3/wav/m4a): Use ParseAudioAction to transcribe and understand audio content.
- For IMAGE: Use ImageAnalysisAction to analyze visual content.
- For WEB SEARCH: Use GoogleSearchAction to find information online.
- For COMPUTATION: Use ExecuteCodeAction to run Python code.
- Always process multimodal inputs FIRST before searching the web, as they contain critical context.

{f"Task Type: {task_type}" if task_type else ""}
{f"Category: {category}" if category else ""}
{f"Multimodal Inputs:" + chr(10) + modal_info if modal_info else ""}

DECISION PROCESS:
1. REVIEW the SUBTASK HISTORY below - check status, result, and key findings of each attempt
2. EVALUATE: Do the results SUFFICIENTLY answer the QUESTION?
   - If any subtask returned a valid result with status "done" → Consider using 'complete'
   - If subtask status is "incomplete" → Review its key findings to see what was accomplished
3. PLAN next action:
   - Results sufficient → Use 'complete' with the answer
   - Need more work → Identify what subtasks are NOW unblocked by previous results
   - Subtask FAILED or INCOMPLETE → You can RETRY the failed/incomplete subtask in the next round. Adjust the instruction, context, or model if needed to improve the chance of success
   - Submit all currently-runnable subtasks in parallel as the next batch (including retries of failed subtasks alongside newly unblocked subtasks)
   - Think ahead: what will you need AFTER this batch? Plan accordingly with your remaining budget

BUDGET AWARENESS:
- You have LIMITED attempts (see Progress below)
- Each delegation (regardless of how many parallel subtasks) counts as ONE attempt
- Maximize parallelism within each round to get the most done per attempt
- Plan your phases wisely: with N remaining attempts, you can run N rounds of parallel subtasks
- If a result looks correct and was verified, trust it and complete

==== MODEL SELECTION GUIDE ====
{model_pricing_table}

Note: Higher-priced models are generally more capable. Price correlates with model strength.

Model Selection Strategy:
- Choose cheaper models for simple tasks (e.g., straightforward web search)
- Choose more capable models for complex reasoning, video analysis, or multi-step tasks
- You can assign DIFFERENT models to different parallel subtasks based on their complexity

==== Progress ====
[Attempt {attempt_index}/{max_attempts}] Remaining {remaining_attempts} attempts
⚠️ Budget is limited. Maximize parallelism to get the most done per attempt.

==== QUESTION ====
{instruction}

==== SUBTASK HISTORY ====
{subtask_history if subtask_history else "No subtasks completed yet."}

==== AVAILABLE TOOLS (for SubAgents) ====
{tools_description}

==== OUTPUT FORMAT ====
ANSWER FORMAT: requires precise, concise answers (single word, number, or short phrase). Do NOT include explanations in the answer field.

Return JSON:

If results are SUFFICIENT:
{{
  "action": "complete",
  "reasoning": "The subtask results show [X], which answers the question",
  "params": {{ "answer": "concise answer" }}
}}

If more work is NEEDED — submit all currently-runnable subtasks in parallel:
{{
  "action": "delegate_task",
  "reasoning": "Based on previous results, [X] and [Y] can now run independently in parallel. [Z] still needs to wait for their results, so I'll handle it in the next round.",
  "params": {{
    "tasks": [
      {{
        "task_instruction": "A SPECIFIC, ACTIONABLE subtask (e.g., 'Analyze the video to identify the main topic discussed')",
        "context": "Relevant findings from previous attempts that this subtask can build on",
        "model": "one of {sub_models}",
        "tools": ["tool1", "tool2"]
      }},
      {{
        "task_instruction": "Another INDEPENDENT subtask that can run at the same time (e.g., 'Search for background information about X')",
        "context": "Relevant context",
        "model": "one of {sub_models}",
        "tools": ["tool3"]
      }}
    ]
  }}
}}

If only ONE subtask can run right now (others depend on its result):
{{
  "action": "delegate_task",
  "reasoning": "I need to first [X] before I can determine [Y]. So this round only has one subtask.",
  "params": {{
    "tasks": [
      {{
        "task_instruction": "The prerequisite subtask that must complete first",
        "context": "Relevant context",
        "model": "one of {sub_models}",
        "tools": ["tool1"]
      }}
    ]
  }}
}}

⚠️ IMPORTANT RULES:
1. ALWAYS use the "tasks" list format (even for a single subtask)
2. Within each round, subtasks must be INDEPENDENT of each other — don't make one subtask depend on another subtask's result IN THE SAME ROUND
3. Subtasks CAN and SHOULD depend on results from PREVIOUS rounds — pass relevant findings via the "context" field
4. Maximize parallelism WITHIN each round: if two things CAN run independently NOW, they SHOULD be parallel subtasks
5. Select relevant tools from AVAILABLE TOOLS section for each subtask
6. Think in phases: what can I do now in parallel? What must wait for next round?
7. If a subtask returns status "failed" or "incomplete", you MAY retry it in the next delegation round. When retrying, consider: adjusting the task instruction to be more specific, providing additional context from other completed subtasks, or switching to a more capable model. Retried subtasks can run in parallel with other new subtasks.
""".strip()
