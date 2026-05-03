"""
OmniGAIA Benchmark Runner with tool-based MainAgent (delegate_task).

Key differences from GAIA Runner:
- Uses OmniGAIA data fields (id, question, answer, Level as string)
- Uses OmniGAIAMainAgentPrompt
- Saves trajectory with multimodal input info
- Uses "omnigaia" benchmark_type
"""
from __future__ import annotations

import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from base.agent.base_action import BaseAction
from base.engine.async_llm import LLMsConfig, create_llm_instance
from base.engine.logs import logger
from benchmark.common.runner import Runner
from benchmark.omnigaia.llm_scorer import llm_semantic_score
from orchestra_o1.main_agent import MainAgent
from orchestra_o1.prompts.omnigaia import OmniGAIAMainAgentPrompt
from orchestra_o1.tools.delegate import DelegateTaskTool


class OmniGAIARunner:
    """Run OmniGAIA levels with a tool-based MainAgent (delegate_task)."""

    def __init__(
        self,
        benchmark,
        main_model: str,
        sub_models: List[str],
        max_attempts: int,
        omnigaia_tools: List[BaseAction],
    ):
        self.benchmark = benchmark
        self.main_model = main_model
        self.sub_models = sub_models
        self.max_attempts = max_attempts
        self.omnigaia_tools = omnigaia_tools

    def _prepare_csv(
        self, csv_path: Path | None
    ) -> Tuple[csv.DictWriter | None, asyncio.Lock | None, Any]:
        if not csv_path:
            return None, None, None
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "task_id",
                "level",
                "task_type",
                "category",
                "main_model",
                "sub_models",
                "final_sub_model",
                "success",
                "reward",
                "attempts",
                "sub_cost",
                "main_cost",
                "total_cost",
                "timestamp",
                "start_time",
                "end_time",
                "error",
            ],
        )
        csv_writer.writeheader()
        csv_file.flush()
        return csv_writer, asyncio.Lock(), csv_file

    def _save_trajectory(
        self,
        trajectory_folder: Path,
        level_id: str,
        timestamp: str,
        level: str | None,
        question: str | None,
        expected_answer: str | None,
        task_type: str | None,
        category: str | None,
        omni_modal_input: list | None,
        instruction: str | None,
        meta: dict | None,
        attempts_detail: List[Dict[str, Any]],
        success: bool,
        total_reward: float,
        total_cost: float,
        main_cost: float,
        sub_cost: float,
        final_sub_model: str | None,
        error: str | None,
        start_time: str | None = None,
        end_time: str | None = None,
        max_attempts: int | None = None,
    ) -> None:
        if not trajectory_folder:
            return
        trajectory_folder.mkdir(parents=True, exist_ok=True)
        filename = f"omnigaia_{level_id}_{timestamp}.json" if timestamp else f"omnigaia_{level_id}.json"
        trajectory_file = trajectory_folder / filename
        data = {
            "task_id": level_id,
            "timestamp": timestamp,
            "start_time": start_time,
            "end_time": end_time,
            "level": level,
            "task_type": task_type,
            "category": category,
            "question": question,
            "expected_answer": expected_answer,
            "omni_modal_input": omni_modal_input,
            "main_model": self.main_model,
            "sub_models": self.sub_models,
            "max_attempts": max_attempts or self.max_attempts,
            "success": success,
            "total_reward": total_reward,
            "total_cost": total_cost,
            "main_cost": main_cost,
            "sub_cost": sub_cost,
            "attempts": len(attempts_detail),
            "trajectory": attempts_detail,
            "final_sub_model": final_sub_model,
            "error": error,
            "instruction": instruction,
            "meta": meta or {},
        }
        with trajectory_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"[OmniGAIA] Trajectory saved to {trajectory_file}")

    async def run_levels(
        self,
        levels,
        max_concurrency: int,
        csv_path: Path | None,
        trajectory_folder: Path,
        timestamp: str,
    ) -> Dict[str, Dict[str, Any]]:
        semaphore = asyncio.Semaphore(max(1, max_concurrency))
        csv_writer, csv_lock, csv_file = self._prepare_csv(csv_path)
        results: Dict[str, Dict[str, Any]] = {}

        async def run_single(level_spec):
            # OmniGAIA uses "id" instead of "task_id"
            level_id = str(level_spec.get("id") or level_spec.get("task_id") or str(level_spec))
            # OmniGAIA Level is a string
            task_level = level_spec.get("Level", "")
            # OmniGAIA uses "question" instead of "Question"
            question = level_spec.get("question", "")
            # OmniGAIA uses "answer" instead of "Final answer"
            expected_answer = level_spec.get("answer", "")
            # OmniGAIA-specific fields
            task_type = level_spec.get("task_type", "")
            category = level_spec.get("category", "")
            omni_modal_input = level_spec.get("omni_modal_input", [])

            async with semaphore:
                env = None
                start_time = datetime.now().isoformat()
                try:
                    logger.info(f"[OmniGAIA] Starting task: {level_id} (Level {task_level}, Type: {task_type})")
                    env = self.benchmark.make_env(level_spec, tools=self.omnigaia_tools)
                    basic_info = env.get_basic_info()

                    main_llm = create_llm_instance(LLMsConfig.default().get(self.main_model))
                    runner = Runner()

                    # Create model alias mapping
                    model_to_alias = {
                        model: f"model_{i+1}" for i, model in enumerate(self.sub_models)
                    }
                    alias_to_model = {v: k for k, v in model_to_alias.items()}

                    delegate_tool = DelegateTaskTool(
                        env=env,
                        runner=runner,
                        models=self.sub_models,
                        benchmark_type="omnigaia",
                        alias_to_model=alias_to_model,
                    )

                    # Use complete action
                    from orchestra_o1.tools.complete import CompleteTool
                    complete_tool = CompleteTool()

                    main_agent = MainAgent(
                        llm=main_llm,
                        sub_models=self.sub_models,
                        tools=[delegate_tool, complete_tool],
                        subagent_tools=self.omnigaia_tools,
                        prompt_builder=OmniGAIAMainAgentPrompt,
                        max_attempts=self.max_attempts,
                        benchmark_type="omnigaia",
                        mask_model_names=True,
                    )

                    # Run MainAgent
                    from benchmark.common.env import BasicInfo
                    main_info = BasicInfo(
                        env_id=level_id,
                        instruction=basic_info.instruction,
                        action_space="",
                        max_steps=self.max_attempts,
                        meta_data=basic_info.meta_data,
                    )
                    main_agent.reset(main_info)

                    main_cost_before = main_agent.get_usage_cost()

                    # Orchestration loop
                    attempts_detail = []
                    final_answer = None

                    for attempt_idx in range(self.max_attempts):
                        action, resp = await main_agent.step(None, [])

                        action_name = action.get("action")
                        params = action.get("params", {})
                        result = action.get("result", {})

                        attempts_detail.append({
                            "attempt": attempt_idx + 1,
                            "action": action_name,
                            "params": params,
                            "result": result,
                            "raw_response": resp,
                        })

                        if action_name == "complete":
                            final_answer = params.get("answer")
                            break

                    main_cost_after = main_agent.get_usage_cost()
                    main_cost = max(0.0, main_cost_after - main_cost_before)

                    # Scoring
                    total_reward = 0.0
                    success = False
                    if final_answer:
                        logger.info(f"[OmniGAIA] MainAgent complete with answer: {final_answer}")

                        # Use LLM semantic evaluation (pass original question for context)
                        try:
                            reward = await llm_semantic_score(str(final_answer), expected_answer, question=question)
                        except Exception as e:
                            logger.warning(f"[OmniGAIA] LLM scoring failed: {e}")
                            reward = 0.0

                        total_reward = reward
                        success = reward > 0.5

                    # Calculate sub_cost (supports parallel subtask result format)
                    sub_cost = 0.0
                    for a in attempts_detail:
                        attempt_result = a.get("result", {})
                        if attempt_result.get("subtask_results"):
                            # Parallel results: accumulate total_cost or each subtask cost
                            sub_cost += float(attempt_result.get("total_cost", 0.0) or 0.0)
                        else:
                            # Backward compatible: old single-task format
                            sub_cost += float(attempt_result.get("cost", 0.0) or 0.0)
                    total_cost = sub_cost + main_cost
                    final_sub_model = None

                    end_time = datetime.now().isoformat()

                    self._save_trajectory(
                        trajectory_folder,
                        level_id,
                        timestamp,
                        task_level,
                        question,
                        expected_answer,
                        task_type,
                        category,
                        omni_modal_input,
                        basic_info.instruction,
                        basic_info.meta_data or {},
                        attempts_detail,
                        success,
                        total_reward,
                        total_cost,
                        main_cost,
                        sub_cost,
                        final_sub_model,
                        None,
                        start_time,
                        end_time,
                    )

                    if csv_writer and csv_lock:
                        async with csv_lock:
                            csv_writer.writerow({
                                "task_id": level_id,
                                "level": task_level,
                                "task_type": task_type,
                                "category": category,
                                "main_model": self.main_model,
                                "sub_models": ",".join(self.sub_models),
                                "final_sub_model": final_sub_model,
                                "success": success,
                                "reward": f"{total_reward:.4f}",
                                "attempts": len(attempts_detail),
                                "sub_cost": f"{sub_cost:.6f}",
                                "main_cost": f"{main_cost:.6f}",
                                "total_cost": f"{total_cost:.6f}",
                                "timestamp": timestamp,
                                "start_time": start_time,
                                "end_time": end_time,
                                "error": None,
                            })
                            csv_file.flush()

                    results[level_id] = {
                        "success": success,
                        "reward": total_reward,
                        "answer": final_answer,
                        "expected_answer": expected_answer,
                    }
                    logger.info(
                        f"[OmniGAIA] Completed task: {level_id} | success={success} "
                        f"reward={total_reward:.4f} attempts={len(attempts_detail)}"
                    )

                except Exception as e:
                    logger.error(f"[OmniGAIA] Task {level_id} failed: {type(e).__name__}: {e}")
                    end_time = datetime.now().isoformat()

                    self._save_trajectory(
                        trajectory_folder,
                        level_id,
                        timestamp,
                        task_level,
                        question,
                        expected_answer,
                        task_type,
                        category,
                        omni_modal_input,
                        None,
                        None,
                        [],
                        False,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        None,
                        str(e),
                        start_time,
                        end_time,
                    )

                    if csv_writer and csv_lock:
                        async with csv_lock:
                            csv_writer.writerow({
                                "task_id": level_id,
                                "level": task_level,
                                "task_type": task_type,
                                "category": category,
                                "main_model": self.main_model,
                                "sub_models": ",".join(self.sub_models),
                                "final_sub_model": None,
                                "success": False,
                                "reward": "0.0000",
                                "attempts": 0,
                                "sub_cost": "0.000000",
                                "main_cost": "0.000000",
                                "total_cost": "0.000000",
                                "timestamp": timestamp,
                                "start_time": start_time,
                                "end_time": end_time,
                                "error": str(e),
                            })
                            csv_file.flush()

                    results[level_id] = {"success": False, "reward": 0.0, "error": str(e)}

                finally:
                    if env and hasattr(env, "close"):
                        try:
                            await env.close()
                        except Exception as e:
                            logger.debug(f"[OmniGAIA] Error closing env for {level_id}: {e}")

        tasks = [asyncio.create_task(run_single(level)) for level in levels]
        try:
            await asyncio.gather(*tasks)
        finally:
            if csv_file:
                csv_file.close()

        return results
