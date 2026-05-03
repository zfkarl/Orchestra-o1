"""DelegateTaskTool - Parallel task delegation tool supporting concurrent SubAgent execution"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List

from pydantic import Field, PrivateAttr

from base.agent.base_action import BaseAction
from base.agent.memory import Memory
from base.engine.async_llm import LLMsConfig, create_llm_instance
from base.engine.logs import logger
from orchestra_o1.subagents import ReActAgent
from orchestra_o1.tools.trace_formatter import create_gaia_formatter


def _make_serializable(obj: Any) -> Any:
    """Recursively convert an object to JSON-serializable format."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _make_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    # Fallback: convert to string
    return str(obj)


class DelegateTaskTool(BaseAction):
    """Parallel task delegation tool supporting concurrent SubAgent executionsubtasks.
    
    MainAgent can submit multiple subtasks (tasks list), each assigned to
    an independent SubAgent with its own env clone for parallel execution.
    In the worst case (single subtask), degrades to serial execution.
    """
    
    name: str = "delegate_task"
    description: str = "Delegate one or more subtasks to SubAgents for parallel execution. Each subtask runs independently with its own SubAgent."
    parameters: Dict[str, Any] = Field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "description": "List of subtasks to execute in parallel. Each subtask is an object with task_instruction, model, context (optional), tools (optional).",
                "items": {
                    "type": "object",
                    "properties": {
                        "task_instruction": {"type": "string", "description": "Task for SubAgent"},
                        "context": {"type": "string", "description": "Additional context/hints"},
                        "model": {"type": "string", "description": "Model to use"},
                        "tools": {"type": "array", "items": {"type": "string"}, "description": "Tools for SubAgent (optional)"},
                    },
                    "required": ["task_instruction", "model"]
                }
            }
        },
        "required": ["tasks"]
    })
    
    # Core dependencies
    env: Any = Field(default=None, exclude=True)
    runner: Any = Field(default=None, exclude=True)
    models: list = Field(default_factory=list)
    
    # Configuration
    benchmark_type: str = Field(default="omnigaia")
    alias_to_model: Dict[str, str] = Field(default_factory=dict)
    
    # Internal state
    _trace_formatter: Any = PrivateAttr(default=None)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(
        self,
        env,
        runner,
        models: list,
        benchmark_type: str = "omnigaia",
        alias_to_model: Dict[str, str] = None,
    ):
        super().__init__()
        self.env = env
        self.runner = runner
        self.models = models
        self.benchmark_type = benchmark_type
        self.alias_to_model = alias_to_model or {}
        
        # # Create trace formatter
        self._trace_formatter = create_gaia_formatter()
        
        # # Set model enum (using aliases or real names)
        display_models = list(self.alias_to_model.keys()) if self.alias_to_model else models
        self.parameters = {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": f"List of subtasks to execute in parallel. Each subtask specifies task_instruction, model (one of {display_models}), optional context, and optional tools.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task_instruction": {"type": "string", "description": "Specific, actionable task for SubAgent"},
                            "context": {"type": "string", "description": "Additional context/hints from previous attempts"},
                            "model": {
                                "type": "string",
                                "description": f"Model to use. MUST be one of: {display_models}",
                                "enum": display_models
                            },
                            "tools": {"type": "array", "items": {"type": "string"}, "description": "Tools for SubAgent (optional)"},
                        },
                        "required": ["task_instruction", "model"]
                    }
                }
            },
            "required": ["tasks"]
        }
    
    async def __call__(
        self,
        tasks: List[Dict[str, Any]] = None,
        # # Backward compatible: supports single-task call format
        task_instruction: str = None,
        model: str = None,
        context: str = "",
        tools: List[str] = None,
    ) -> Dict:
        """Execute delegated tasks (supports parallel multi-task)
        
        Args:
            tasks: List of subtasks, each containing task_instruction, model, context, tools
            task_instruction: (backward compatible) single task instruction
            model: (backward compatible) single task model
            context: (backward compatible) single task context
            tools: (backward compatible) single task tool list
            
        Returns:
            Dictionary containing all parallel subtask execution results
        """
        # # Backward compatible: convert old single-task format to tasks list
        if tasks is None:
            if task_instruction and model:
                tasks = [{
                    "task_instruction": task_instruction,
                    "model": model,
                    "context": context,
                    "tools": tools,
                }]
            else:
                return {"error": "Must provide either 'tasks' list or 'task_instruction'+'model'", "subtask_results": []}
        
        if not tasks:
            return {"error": "Empty tasks list", "subtask_results": []}
        
        logger.info(f"[DelegateTool] Launching {len(tasks)} parallel subtask(s)")
        
        # # Execute all subtasks in parallel
        coroutines = [self._execute_single_task(task, idx) for idx, task in enumerate(tasks)]
        subtask_results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # # Handle exception results
        processed_results = []
        for idx, result in enumerate(subtask_results):
            if isinstance(result, Exception):
                logger.error(f"[DelegateTool] Subtask {idx} failed with exception: {result}")
                processed_results.append({
                    "subtask_index": idx,
                    "task_instruction": tasks[idx].get("task_instruction", ""),
                    "error": str(result),
                    "steps_taken": 0,
                    "done": False,
                    "cost": 0.0,
                })
            else:
                result["subtask_index"] = idx
                processed_results.append(result)
        
        # # Aggregate statistics
        total_cost = sum(r.get("cost", 0.0) or 0.0 for r in processed_results)
        total_steps = sum(r.get("steps_taken", 0) for r in processed_results)
        all_done = all(r.get("done", False) for r in processed_results)
        
        return {
            "subtask_results": processed_results,
            "total_subtasks": len(tasks),
            "total_cost": total_cost,
            "total_steps": total_steps,
            "all_done": all_done,
        }
    
    async def _execute_single_task(self, task: Dict[str, Any], idx: int) -> Dict:
        """Execute a single subtask (using an independent env clone)
        
        Args:
            task: Subtask configuration {task_instruction, model, context, tools}
            idx: Subtask index
            
        Returns:
            Subtask execution result
        """
        task_instruction = task.get("task_instruction", "")
        model = task.get("model", "")
        context = task.get("context", "")
        tools = task.get("tools")
        
        # 1. # Resolve model name
        real_model = self.alias_to_model.get(model, model)
        if real_model not in self.models:
            return {"error": f"Invalid model: {model}", "steps_taken": 0, "done": False, "cost": 0.0, "task_instruction": task_instruction}
        
        logger.info(f"[DelegateTool] Subtask {idx}: model={real_model}, tools={tools}, task={task_instruction[:80]}...")
        
        # 2. # Clone independent env instance (avoid shared state conflicts during parallel execution)
        env_clone = self.env.clone()
        
        # 3. # Get original question
        original_question = getattr(self.env, 'instruction', '') or ''
        
        # 4. # Create SubAgent
        llm = create_llm_instance(LLMsConfig.default().get(real_model))
        
        sub_agent = ReActAgent(
            llm=llm,
            benchmark_type=self.benchmark_type,
            task_instruction=task_instruction,
            context=context,
            original_question=original_question,
            allowed_tools=tools,
            memory=Memory(llm=llm, max_memory=10),
        )
        
        # 5. # Set cloned env instruction
        if hasattr(env_clone, 'instruction'):
            env_clone.instruction = task_instruction
        
        try:
            # 6. # Execute (using cloned env)
            result = await self.runner.run(sub_agent, env_clone)
            
            # 7. # Extract finish_result
            finish_result = None
            if result.trace:
                last = result.trace[-1]
                if last.info.get("finished") and last.info.get("finish_result"):
                    finish_result = last.info["finish_result"]
            
            # 8. # Summarize trace
            trace_summary = await self._summarize_trace(result.trace, task_instruction)
            
            # # Serialize trace
            trace_serializable = [_make_serializable(step) for step in result.trace] if result.trace else []
            
            return {
                "task_instruction": task_instruction,
                "model": real_model,
                "tools_assigned": tools,
                "steps_taken": result.steps,
                "done": result.done,
                "cost": result.cost,
                "finish_result": finish_result,
                "trace": trace_serializable,
                "trace_summary": trace_summary,
                "statistics": {
                    "total_steps": result.steps,
                    "max_steps": 30,
                    "completed": result.done
                },
            }
            
        except Exception as e:
            logger.error(f"[DelegateTool] Subtask {idx} error: {e}")
            return {"error": str(e), "task_instruction": task_instruction, "steps_taken": 0, "done": False, "cost": 0.0}
        
        finally:
            # 9. # Close cloned env
            if hasattr(env_clone, 'close'):
                try:
                    await env_clone.close()
                except Exception:
                    pass
    
    async def _summarize_trace(self, trace, task_instruction: str) -> str:
        """Summarize execution trace (using gpt-5 model)."""
        if not trace:
            return "No steps executed"
        
        trace_text = self._trace_formatter.format_trace(trace)
        
        prompt = f"""You are a trajectory summarizer. Review the SubAgent's execution trace.

Task: {task_instruction[:200]}
Steps: {len(trace)}

=== Trace ===
{trace_text}
===

Summarize in 5-10 bullets: key progress, problems, remaining issues.
Output ONLY bullets."""
        
        try:
            review_llm = create_llm_instance(
                LLMsConfig.default().get("gpt-5")
            )
            return (await review_llm(prompt)).strip()
        except Exception as e:
            logger.warning(f"[DelegateTool] Trace summarization failed: {e}")
            return f"Steps: {len(trace)}"
