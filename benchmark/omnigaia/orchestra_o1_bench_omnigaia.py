"""
OmniGAIA benchmark - Orchestra mode (MainAgent + SubAgent).

Key differences from GAIA:
1. Data fields: id (numeric), question, answer, Level (Easy/Medium/Hard), omni_modal_input (multimodal)
2. Attachment format: omni_modal_input is a list with type (video/audio/image), id, path per item
3. Attachment directory structure: images/, videos/, audios/ subdirectories
4. Video analysis capability via VideoAnalysisAction

This module provides:
- OmniGAIAOrchestraEnvironment: SubAgent uses 'finish' to report results to MainAgent
- OmniGAIAOrchestraBenchmark: Factory for creating Orchestra environments
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from base.agent.base_action import BaseAction
from base.engine.logs import logger
from benchmark.benchmark import Benchmark, LevelSpec
from benchmark.common.env import Action, BasicInfo, Environment, Observation
from benchmark.omnigaia.llm_scorer import llm_semantic_score


PROJECT_ROOT = Path(__file__).parent.parent.parent

# Mapping from OmniGAIA multimodal types to tool hints
OMNI_MODAL_TOOL_HINTS = {
    "video": "Use VideoAnalysisAction to analyze video content (visual frames + audio track). It automatically detects and transcribes audio when present. Set analyze_audio=true for videos with speech/sound. You can specify start_time/end_time to focus on specific segments.",
    "audio": "Use ParseAudioAction to transcribe and analyze audio content.",
    "image": "Use ImageAnalysisAction to analyze image content.",
}

# Mapping from file extensions to tool hints (for attachments)
FILE_TOOL_HINTS = {
    '.png': "Use ImageAnalysisAction to analyze this image.",
    '.jpg': "Use ImageAnalysisAction to analyze this image.",
    '.jpeg': "Use ImageAnalysisAction to analyze this image.",
    '.gif': "Use ImageAnalysisAction to analyze this image.",
    '.webp': "Use ImageAnalysisAction to analyze this image.",
    '.mp3': "Use ParseAudioAction to transcribe this audio file.",
    '.wav': "Use ParseAudioAction to transcribe this audio file.",
    '.m4a': "Use ParseAudioAction to transcribe this audio file.",
    '.ogg': "Use ParseAudioAction to transcribe this audio file.",
    '.mp4': "Use VideoAnalysisAction to analyze this video (supports both visual frames and audio track extraction).",
    '.avi': "Use VideoAnalysisAction to analyze this video (supports both visual frames and audio track extraction).",
    '.mov': "Use VideoAnalysisAction to analyze this video (supports both visual frames and audio track extraction).",
    '.mkv': "Use VideoAnalysisAction to analyze this video (supports both visual frames and audio track extraction).",
    '.webm': "Use VideoAnalysisAction to analyze this video (supports both visual frames and audio track extraction).",
    '.xlsx': "Use ExecuteCodeAction with pandas to read and analyze this spreadsheet.",
    '.csv': "Use ExecuteCodeAction with pandas to read and analyze this spreadsheet.",
    '.pdf': "Use ExecuteCodeAction with appropriate libraries to extract text from this document.",
    '.docx': "Use ExecuteCodeAction with appropriate libraries to extract text from this document.",
    '.pptx': "Use ExecuteCodeAction with appropriate libraries to extract text from this document.",
    '.py': "Use ExecuteCodeAction to run or analyze this Python script.",
    '.txt': "Use ExecuteCodeAction to read and process this text/JSON file.",
    '.json': "Use ExecuteCodeAction to read and process this text/JSON file.",
}

# Action space template for OmniGAIA Orchestra mode
ACTION_SPACE_TEMPLATE = """
### finish
Description: Report your result to MainAgent. Use when you have found the answer or cannot proceed.
Parameters: {{"result": "<answer>", "status": "done|partial|blocked", "summary": "<brief summary of what you did>"}}

[IMPORTANT RULES]
- Use print() in ExecuteCodeAction to see computation results
- Use 'finish' to report your result when done
- When remaining steps < 5, finish with your best result
- For video analysis, use VideoAnalysisAction with analyze_audio=true to extract both visual frames and audio track. Use start_time/end_time to focus on specific segments.
- For standalone audio files, use ParseAudioAction to transcribe and analyze audio
- For image analysis, use ImageAnalysisAction

[STRICT OUTPUT FORMAT]
⚠️ You MUST output a single JSON object. The "action" field MUST be one of the EXACT tool names above (e.g., "ImageAnalysisAction") or "finish".
⚠️ Do NOT use "execute" as the action name. Do NOT wrap tool names inside a "command" field.
⚠️ The "params" field MUST contain the tool's parameter keys as a JSON object — NOT as command-line flags.

General format:
{{"action": "<EXACT_TOOL_NAME>", "params": {{<key>: <value>, ...}}, "memory": "<observations>"}}

[FEW-SHOT EXAMPLES FOR EACH TOOL]

GoogleSearchAction:
{{"action": "GoogleSearchAction", "params": {{"query": "capital of France"}}, "memory": "Searching for the capital."}}

ImageAnalysisAction:
{{"action": "ImageAnalysisAction", "params": {{"query": "What brand logo is shown?", "image_path": "/data/images/photo.jpg"}}, "memory": "Identifying brand in image."}}

ParseAudioAction:
{{"action": "ParseAudioAction", "params": {{"query": "Transcribe and identify the main topic discussed.", "audio_path": "/data/audios/clip.wav"}}, "memory": "Transcribing audio."}}

VideoAnalysisAction:
{{"action": "VideoAnalysisAction", "params": {{"query": "What text appears on screen at 30s?", "video_path": "/data/videos/clip.mp4", "max_frames": 8, "start_time": 25, "end_time": 35, "analyze_audio": true}}, "memory": "Analyzing 25-35s segment."}}

ExecuteCodeAction:
{{"action": "ExecuteCodeAction", "params": {{"code": "import math\\nresult = math.sqrt(144)\\nprint(result)", "code_type": "python"}}, "memory": "Computing square root."}}

ExtractUrlContentAction:
{{"action": "ExtractUrlContentAction", "params": {{"url": "https://example.com/page", "browse_query": "key facts"}}, "memory": "Fetching web page."}}

finish:
{{"action": "finish", "params": {{"result": "42", "status": "done", "summary": "Found the answer via video analysis."}}, "memory": "Task complete."}}
""".strip()


@dataclass
class OmniGAIAConfig:
    """Configuration for the OmniGAIA benchmark."""
    dataset_path: Path
    attachments_dir: Path
    max_steps: int = 30
    max_tasks: Optional[int] = None
    level_filter: Optional[List[str]] = None  # String levels: "Easy"/"Medium"/"Hard"
    result_folder: Path = PROJECT_ROOT / "workspace/logs/omnigaia_results"
    trajectory_folder: Path = PROJECT_ROOT / "workspace/logs/omnigaia_trajectories"
    timestamp: Optional[str] = None

    @classmethod
    def load(cls, config_path: Path | str) -> "OmniGAIAConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        dataset_path = raw.get("dataset_path")
        if dataset_path:
            dataset_path = PROJECT_ROOT / dataset_path if not Path(dataset_path).is_absolute() else Path(dataset_path)
        else:
            raise ValueError("dataset_path is required for OmniGAIA")

        attachments_dir = raw.get("attachments_dir")
        if attachments_dir:
            attachments_dir = PROJECT_ROOT / attachments_dir if not Path(attachments_dir).is_absolute() else Path(attachments_dir)
        else:
            raise ValueError("attachments_dir is required for OmniGAIA")

        level_filter = raw.get("level_filter")
        if level_filter is not None:
            if isinstance(level_filter, str):
                level_filter = [level_filter]
            else:
                level_filter = [str(l) for l in level_filter]

        def resolve_path(key, default):
            path = raw.get(key, default)
            return PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)

        return cls(
            dataset_path=Path(dataset_path),
            attachments_dir=Path(attachments_dir),
            max_steps=int(raw.get("max_steps", 30)),
            max_tasks=int(raw["max_tasks"]) if raw.get("max_tasks") else None,
            level_filter=level_filter,
            result_folder=resolve_path("result_folder", "workspace/logs/omnigaia_results"),
            trajectory_folder=resolve_path("trajectory_folder", "workspace/logs/omnigaia_trajectories"),
        )


class OmniGAIAOrchestraEnvironment(Environment):
    """
    Environment for Orchestra-mode OmniGAIA tasks.

    SubAgent uses the 'finish' action to report results back to MainAgent.
    Supports multimodal inputs (omni_modal_input: video/audio/image),
    resolves multimodal attachment paths, and builds instructions
    that include multimodal context.
    """

    def __init__(self, level: LevelSpec, config: OmniGAIAConfig, tools: List[BaseAction]):
        # OmniGAIA uses "id" instead of "task_id"
        self.task_id = str(level.get("id") or level.get("task_id") or "unknown")
        self.config = config
        self.tools: Dict[str, BaseAction] = {t.name: t for t in tools}

        # OmniGAIA uses "question" instead of "Question"
        self.question = level.get("question") or level.get("Question") or str(level)
        # OmniGAIA uses "answer" instead of "Final answer"
        self.expected_answer = level.get("answer") or level.get("Final answer")
        # OmniGAIA Level is a string: "Easy"/"Medium"/"Hard"
        self.task_level = level.get("Level") or level.get("level")

        # OmniGAIA multimodal inputs
        self.omni_modal_input = level.get("omni_modal_input", [])
        self.resolved_modal_inputs = self._resolve_modal_inputs(config.attachments_dir)

        # OmniGAIA additional metadata
        self.task_type = level.get("task_type", "")
        self.category = level.get("category", "")
        self.required_external_tools = level.get("required_external_tools", [])
        self.total_steps = level.get("total_steps")

        self.level_data = level
        self.meta_data = {
            "task_id": self.task_id,
            "level": self.task_level,
            "task_type": self.task_type,
            "category": self.category,
            "required_external_tools": self.required_external_tools,
            "modal_inputs": [
                {"type": m["type"], "id": m.get("id", ""), "path": str(m.get("resolved_path", ""))}
                for m in self.resolved_modal_inputs
            ],
            "expected_steps": self.total_steps,
        }

        self._steps = 0
        self._done = False

    def clone(self) -> "OmniGAIAOrchestraEnvironment":
        """Create an independent environment clone for parallel SubAgent execution.
        
        Each parallel SubAgent needs an independent env instance to avoid shared state conflicts
        (e.g. _steps, _done, instruction, etc.).
        """
        from copy import deepcopy
        # Use level_data to construct a fresh env instance
        tools_list = list(self.tools.values())
        new_env = OmniGAIAOrchestraEnvironment(
            level=self.level_data,
            config=self.config,
            tools=tools_list,
        )
        return new_env

    def _resolve_modal_inputs(self, attachments_dir: Path) -> list:
        """Resolve and validate multimodal input file paths.

        OmniGAIA omni_modal_input format:
        [{"type": "video", "id": "video_xxx", "path": "videos/xxx.mp4"}, ...]
        """
        resolved = []
        for modal_item in self.omni_modal_input:
            item = dict(modal_item)
            rel_path = item.get("path", "")
            if rel_path:
                full_path = attachments_dir / rel_path
                if full_path.exists():
                    item["resolved_path"] = full_path
                else:
                    logger.warning(
                        f"[OmniGAIA Orchestra] Modal file not found: {full_path} "
                        f"(task {self.task_id}, type={item.get('type')})"
                    )
                    item["resolved_path"] = None
            else:
                item["resolved_path"] = None
            resolved.append(item)
        return resolved

    def _build_action_space(self) -> str:
        """Build action space description for SubAgent."""
        tool_descriptions = []
        for name, tool in self.tools.items():
            desc = f"### {name}\nDescription: {tool.description}"
            if tool.parameters:
                desc += f"\nParameters: {json.dumps(tool.parameters, indent=2)}"
            tool_descriptions.append(desc)
        return "Available actions:\n\n" + "\n\n".join(tool_descriptions) + "\n\n" + ACTION_SPACE_TEMPLATE

    def _build_instruction(self) -> str:
        """Build instruction including the question and multimodal input hints.

        OmniGAIA provides a multimodal input list; each input gets a tool hint.
        """
        instruction = f"Question: {self.question}"

        # Add multimodal input hints
        if self.resolved_modal_inputs:
            instruction += "\n\n[MULTIMODAL INPUTS]"
            for i, modal in enumerate(self.resolved_modal_inputs, 1):
                modal_type = modal.get("type", "unknown")
                modal_id = modal.get("id", "")
                resolved_path = modal.get("resolved_path")

                if resolved_path:
                    tool_hint = OMNI_MODAL_TOOL_HINTS.get(modal_type, "Use appropriate tool to process this file.")
                    instruction += f"\n{i}. [{modal_type.upper()}] ID: {modal_id}"
                    instruction += f"\n   Path: {resolved_path}"
                    instruction += f"\n   Hint: {tool_hint}"
                else:
                    instruction += f"\n{i}. [{modal_type.upper()}] ID: {modal_id} (file not available)"

        return instruction

    def get_basic_info(self) -> BasicInfo:
        """Get basic information about the task."""
        return BasicInfo(
            env_id=self.task_id,
            instruction=self._build_instruction(),
            action_space=self._build_action_space(),
            max_steps=self.config.max_steps,
            meta_data=self.meta_data,
        )

    async def reset(self, seed: int | None = None) -> Observation:
        """Reset environment."""
        self._done = False
        self._steps = 0
        return {
            "message": "Environment ready. Use the available tools to answer the question.",
            "question": self.question,
            "current_step": 0,
            "max_steps": self.config.max_steps,
        }

    async def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute action and return observation."""
        if self._done:
            raise RuntimeError("Environment already finished. Call reset() first.")

        self._steps += 1
        action_type = action.get("action", "")
        params = action.get("params", {})

        if action_type == "finish":
            return self._handle_finish(params)
        if action_type == "SubmitAnswer":
            return self._handle_submit(params)

        return await self._handle_tool(action_type, params)

    def _handle_finish(self, params: Dict) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Handle SubAgent finish — report to MainAgent without scoring."""
        result = params.get("result", "")
        status = params.get("status", "done")
        summary = params.get("summary", "")

        self._done = True
        finish_result = {"result": result, "status": status, "summary": summary}

        logger.info(f"[OmniGAIA Orchestra] Task {self.task_id} finish: result='{result}', status='{status}'")

        return {
            "message": "Result reported to MainAgent.",
            "current_step": self._steps,
            "finish_result": finish_result,
        }, 0.0, True, {"finished": True, "finish_result": finish_result}

    def _handle_submit(self, params: Dict) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Handle SubmitAnswer — runner triggers final scoring."""
        answer = params.get("answer", "")
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        llm_semantic_score(answer, self.expected_answer, question=self.question)
                    )
                    reward = future.result(timeout=30)
            else:
                reward = loop.run_until_complete(
                    llm_semantic_score(answer, self.expected_answer, question=self.question)
                )
        except Exception:
            reward = 0.0
        self._done = True

        logger.info(
            f"[OmniGAIA Orchestra] Task {self.task_id} submitted: "
            f"answer='{answer}', expected='{self.expected_answer}', reward={reward}"
        )

        return {
            "message": "Answer submitted",
            "submitted_answer": answer,
            "expected_answer": self.expected_answer,
            "reward": reward,
            "correct": reward == 1.0,
            "current_step": self._steps,
        }, reward, True, {"submitted": True, "correct": reward == 1.0}

    async def _handle_tool(self, action_type: str, params: Dict) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Handle tool execution."""
        tool = self.tools.get(action_type)

        if tool is None:
            return self._handle_unknown_action(action_type)

        try:
            result = await tool(**params)
            observation = {
                "action": action_type,
                "success": result.get("success", False),
                "output": result.get("output") if result.get("success") else None,
                "error": result.get("error") if not result.get("success") else None,
                "current_step": self._steps,
                "max_steps": self.config.max_steps,
            }
            logger.info(
                f"[OmniGAIA Orchestra] Task {self.task_id} step {self._steps}: "
                f"{action_type} -> success={result.get('success')}"
            )
        except Exception as e:
            observation = {
                "action": action_type,
                "success": False,
                "error": str(e),
                "current_step": self._steps,
                "max_steps": self.config.max_steps,
            }
            logger.error(f"[OmniGAIA Orchestra] Task {self.task_id} tool execution error: {e}")

        return self._check_max_steps(observation)

    def _handle_unknown_action(self, action_type: str) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Handle unknown action type."""
        observation = {
            "error": f"Unknown action: {action_type}. Available actions: {list(self.tools.keys()) + ['finish']}",
            "current_step": self._steps,
            "max_steps": self.config.max_steps,
        }

        if self._steps >= self.config.max_steps:
            return self._timeout_response(observation, {"error": "unknown_action"})

        return observation, 0.0, False, {"error": "unknown_action"}

    def _check_max_steps(self, observation: Dict) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Check if max steps reached."""
        if self._steps >= self.config.max_steps:
            return self._timeout_response(observation, {})
        return observation, 0.0, False, {}

    def _timeout_response(self, observation: Dict, extra_info: Dict) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Generate timeout response when max steps reached."""
        self._done = True
        finish_result = {
            "result": "",
            "status": "timeout",
            "summary": f"Used all {self.config.max_steps} steps without finish",
        }
        observation["message"] = "Max steps reached"
        observation["finish_result"] = finish_result
        return observation, 0.0, True, {
            **extra_info,
            "max_steps_reached": True,
            "finished": True,
            "finish_result": finish_result,
        }

    async def close(self):
        """Clean up environment resources."""
        pass


class OmniGAIAOrchestraBenchmark(Benchmark):
    """
    OmniGAIA Benchmark for Orchestra mode (MainAgent + SubAgent).

    Creates OmniGAIAOrchestraEnvironment instances where SubAgent uses 'finish'.
    Parses the OmniGAIA data format (id, question, answer, Level string,
    omni_modal_input) with string-based level filtering.
    """

    def __init__(self, config: OmniGAIAConfig, tools: List[BaseAction] | None = None):
        self.config = config
        self.tools = tools or []
        self._levels: List[LevelSpec] = []
        self._load_dataset()

    def _load_dataset(self):
        """Load OmniGAIA dataset from JSONL file."""
        if not self.config.dataset_path.exists():
            logger.warning(f"[OmniGAIA Orchestra] Dataset not found: {self.config.dataset_path}")
            return

        self._levels = []
        with self.config.dataset_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not (line := line.strip()):
                    continue
                try:
                    data = json.loads(line)
        # OmniGAIA uses numeric id; convert to string for consistency
                    if "task_id" not in data and "id" in data:
                        data["task_id"] = str(data["id"])
                    elif "task_id" not in data and "id" not in data:
                        data["task_id"] = f"omnigaia_task_{line_num}"

                    # Apply level_filter (OmniGAIA Level is a string)
                    if self.config.level_filter is not None:
                        task_level = data.get("Level", "")
                        if task_level not in self.config.level_filter:
                            continue

                    self._levels.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"[OmniGAIA Orchestra] Failed to parse line {line_num}: {e}")

        # Summary statistics
        level_counts = {}
        for level in self._levels:
            l = level.get("Level", "unknown")
            level_counts[l] = level_counts.get(l, 0) + 1

        type_counts = {}
        for level in self._levels:
            t = level.get("task_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        with_modal = sum(1 for l in self._levels if l.get("omni_modal_input"))
        logger.info(f"[OmniGAIA Orchestra] Loaded {len(self._levels)} tasks from {self.config.dataset_path}")
        logger.info(f"[OmniGAIA Orchestra] Level distribution: {level_counts}")
        logger.info(f"[OmniGAIA Orchestra] Task type distribution: {type_counts}")
        logger.info(f"[OmniGAIA Orchestra] With multimodal inputs: {with_modal}")

    def list_levels(self) -> List[LevelSpec]:
        """Return list of all levels/tasks."""
        levels = self._levels
        if self.config.max_tasks and len(levels) > self.config.max_tasks:
            levels = levels[:self.config.max_tasks]
        return levels

    def make_env(self, level: LevelSpec, tools: List[BaseAction] | None = None) -> OmniGAIAOrchestraEnvironment:
        """Create OmniGAIAOrchestraEnvironment for a specific level."""
        return OmniGAIAOrchestraEnvironment(level, self.config, tools if tools is not None else self.tools)

    def get_level_by_id(self, task_id: str) -> Optional[LevelSpec]:
        """Get a specific task by its ID."""
        return next(
            (l for l in self._levels if str(l.get("task_id")) == task_id or str(l.get("id")) == task_id),
            None,
        )
