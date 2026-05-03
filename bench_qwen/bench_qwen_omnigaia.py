"""
OmniGAIA Benchmark with Orchestra-o1 - Qwen3-8B as MainAgent

Uses locally deployed Qwen3-8B as MainAgent (task decomposition and coordination),
SubAgent, Trace Summary, Scorer etc. still use commercial models.

Usage:
    1. Start vLLM server:
       bash bench_qwen/start_vllm.sh

    2. Run evaluation:
       python bench_qwen/bench_qwen_omnigaia.py
       
       # Examples:
       python bench_qwen/bench_qwen_omnigaia.py --config bench_qwen/orchestra_o1_omnigaia_qwen.yaml
       python bench_qwen/bench_qwen_omnigaia.py --max_concurrency 3
       python bench_qwen/bench_qwen_omnigaia.py --tasks "1,2,3"
       python bench_qwen/bench_qwen_omnigaia.py --skip_completed logs/omnigaia_qwen/xxx.csv
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

# ==================== Network proxy (configure if needed) ====================
# import os as _os
# _os.environ.setdefault("http_proxy", "http://your-proxy:port")
# _os.environ.setdefault("https_proxy", "http://your-proxy:port")
# del _os
# ======================================================

import argparse
import asyncio
import csv
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.engine.logs import logger
from base.engine.async_llm import LLMsConfig
from benchmark.omnigaia.orchestra_o1_bench_omnigaia import OmniGAIAOrchestraBenchmark, OmniGAIAConfig
from benchmark.omnigaia.tools import (
    GoogleSearchAction,
    ExecuteCodeAction,
    ExtractUrlContentAction,
    ImageAnalysisAction,
    ParseAudioAction,
    VideoAnalysisAction,
)
from orchestra_o1.config import GAIAOrchestraConfig
from orchestra_o1.runners.omnigaia_runner import OmniGAIARunner

# ==================== aiohttp  ====================
# aiohttp default,  monkey-patch
#  aiohttp  https_proxy / http_proxy , 
# , .
import aiohttp as _aiohttp

_AIOHTTP_PROXY = os.getenv("https_proxy", os.getenv("HTTPS_PROXY", "")) or \
                 os.getenv("http_proxy", os.getenv("HTTP_PROXY", ""))

if _AIOHTTP_PROXY:
    _original_request = _aiohttp.ClientSession._request

    async def _patched_request(self, method, url, **kwargs):
        # call proxy, 
        if "proxy" not in kwargs or kwargs["proxy"] is None:
            kwargs["proxy"] = _AIOHTTP_PROXY
        return await _original_request(self, method, url, **kwargs)

    _aiohttp.ClientSession._request = _patched_request
    logger_msg = f"[Qwen Bench] Injected proxy for aiohttp: {_AIOHTTP_PROXY}"
else:
    logger_msg = "[Qwen Bench] No proxy env detected, aiohttp will connect directly"
# : logger ,  main() 
# ===========================================================

# default
BENCH_QWEN_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BENCH_QWEN_DIR / "orchestra_o1_omnigaia_qwen.yaml"
DEFAULT_MODEL_CONFIG_PATH = BENCH_QWEN_DIR / "model_config_qwen.yaml"


def load_custom_model_config(model_config_path: Path) -> None:
    """Load custom model config (including qwen3-8b local vLLM config).
    
    Overrides the LLMsConfig default singleton so that all subsequent
    LLMsConfig.default() calls use the config containing qwen3-8b.
    """
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config file not found: {model_config_path}")
    
    with open(model_config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}
    
    if "models" in config_data:
        config_data = config_data["models"] or {}
    
    #  LLMsConfig 
    LLMsConfig._default_config = LLMsConfig(config_data)
    
    model_names = list(config_data.keys())
    logger.info(f"[Qwen Bench] Loaded custom model config, available models: {model_names}")


async def main():
    """Run OmniGAIA benchmark with Qwen3-8B as MainAgent."""
    parser = argparse.ArgumentParser(
        description="Run OmniGAIA benchmark using Qwen3-8B as MainAgent."
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH),
        help="Path to benchmark config YAML."
    )
    parser.add_argument(
        "--model_config", default=str(DEFAULT_MODEL_CONFIG_PATH),
        help="Path to model config YAML (contains qwen3-8b vLLM endpoint)."
    )
    parser.add_argument(
        "--max_concurrency", type=int, default=10,
        help="Override max_concurrency."
    )
    parser.add_argument(
        "--tasks", type=str, default=None,
        help="Comma-separated task IDs."
    )    
    parser.add_argument(
        "--skip_completed", type=str, default=None,
        help="Path to existing CSV to skip completed tasks."
    )
    parser.add_argument(
        "--vllm_url", type=str, default=None,
        help="Override vLLM server URL (e.g., http://localhost:8801/v1/)."
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("OmniGAIA Benchmark with Orchestra-o1 (Qwen3-8B MainAgent)")
    logger.info("=" * 60)
    logger.info(logger_msg)  # Print aiohttp proxy patch status

    # 1. Load model config
    model_config_path = Path(args.model_config)
    load_custom_model_config(model_config_path)

    #  vllm_url,  qwen3-8b  base_url
    if args.vllm_url:
        llms_config = LLMsConfig.default()
        if "qwen3-8b" in llms_config.configs:
            llms_config.configs["qwen3-8b"]["base_url"] = args.vllm_url
            logger.info(f"[Qwen Bench] Updated qwen3-8b base_url -> {args.vllm_url}")
        else:
            logger.warning("[Qwen Bench] qwen3-8b not found in model config, --vllm_url ignored")

    # 2. Load benchmark config
    cfg = GAIAOrchestraConfig.load(args.config)

    # Check dataset
    if not cfg.dataset_path.exists():
        logger.error(f"Dataset not found: {cfg.dataset_path}")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.timestamp = timestamp

    logger.info(f"MainAgent Model: {cfg.main_model}")
    logger.info(f"SubAgent Models: {cfg.sub_models}")
    logger.info(f"Dataset: {cfg.dataset_path}")
    logger.info(f"Attachments: {cfg.attachments_dir}")
    logger.info(f"Level filter: {cfg.level_filter}")

    # 3. Initialize OmniGAIA tools
    omnigaia_tools = [
        GoogleSearchAction(),
        ExecuteCodeAction(),
        ExtractUrlContentAction(),
        ImageAnalysisAction(),
        ParseAudioAction(),
        VideoAnalysisAction(),
    ]
    logger.info(f"Loaded {len(omnigaia_tools)} OmniGAIA tools: {[t.name for t in omnigaia_tools]}")

    # 4. Create benchmark
    omnigaia_cfg = OmniGAIAConfig(
        dataset_path=cfg.dataset_path,
        attachments_dir=cfg.attachments_dir,
        level_filter=[str(l) for l in cfg.level_filter] if cfg.level_filter else None,
        max_tasks=cfg.max_tasks,
        result_folder=cfg.result_folder,
        trajectory_folder=cfg.trajectory_folder,
    )
    benchmark = OmniGAIAOrchestraBenchmark(omnigaia_cfg, tools=omnigaia_tools)
    levels = benchmark.list_levels()

    if not levels:
        logger.error("No tasks found in dataset!")
        return 1

    # Filter by task IDs
    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
        levels = [l for l in levels if str(l.get("id") or l.get("task_id")) in task_ids]
        logger.info(f"Filtered to {len(levels)} task(s)")
    elif cfg.max_tasks and len(levels) > cfg.max_tasks:
        levels = levels[:cfg.max_tasks]
        logger.info(f"Limited to {len(levels)} task(s)")

    # Skip completed
    if args.skip_completed:
        skip_csv_path = Path(args.skip_completed)
        if skip_csv_path.exists():
            completed_task_ids = set()
            with open(skip_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    task_id = row.get("task_id") or row.get("id")
                    if task_id:
                        completed_task_ids.add(task_id)

            original_count = len(levels)
            levels = [l for l in levels if str(l.get("id") or l.get("task_id")) not in completed_task_ids]
            skipped_count = original_count - len(levels)
            logger.info(f"Skipped {skipped_count} completed task(s)")

    logger.info(f"Running {len(levels)} task(s)")

    # 5. Create output directories
    cfg.result_folder.mkdir(parents=True, exist_ok=True)
    cfg.trajectory_folder.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.result_folder / f"omnigaia_qwen_{timestamp}.csv"

    # 6. Create runner
    runner = OmniGAIARunner(
        benchmark=benchmark,
        main_model=cfg.main_model,
        sub_models=cfg.sub_models,
        max_attempts=cfg.max_attempts,
        omnigaia_tools=omnigaia_tools,
    )

    logger.info(f"[Qwen Bench] main_model={cfg.main_model}, sub_models={cfg.sub_models}")
    logger.info(f"Max concurrency: {args.max_concurrency or cfg.max_concurrency}")
    logger.info(f"Results: {csv_path}")

    # 7. Run benchmark
    results = await runner.run_levels(
        levels=levels,
        max_concurrency=args.max_concurrency or cfg.max_concurrency,
        csv_path=csv_path,
        trajectory_folder=cfg.trajectory_folder,
        timestamp=timestamp,
    )

    # 8. Print summary
    total = len(results)
    success_count = sum(1 for r in results.values() if r.get("success"))
    total_reward = sum(float(r.get("reward", 0) or 0) for r in results.values())

    # Statistics by level
    level_stats = {}
    for level_spec in levels:
        lid = str(level_spec.get("id") or level_spec.get("task_id"))
        lvl = level_spec.get("Level", "unknown")
        if lid in results:
            if lvl not in level_stats:
                level_stats[lvl] = {"total": 0, "success": 0}
            level_stats[lvl]["total"] += 1
            if results[lid].get("success"):
                level_stats[lvl]["success"] += 1

    # Statistics by task_type
    type_stats = {}
    for level_spec in levels:
        lid = str(level_spec.get("id") or level_spec.get("task_id"))
        ttype = level_spec.get("task_type", "unknown")
        if lid in results:
            if ttype not in type_stats:
                type_stats[ttype] = {"total": 0, "success": 0}
            type_stats[ttype]["total"] += 1
            if results[lid].get("success"):
                type_stats[ttype]["success"] += 1

    logger.info("\n" + "=" * 60)
    logger.info("OmniGAIA Benchmark Summary (Qwen3-8B MainAgent):")
    logger.info(f"  Main Model: {cfg.main_model}")
    logger.info(f"  Sub Models: {cfg.sub_models}")
    logger.info(f"  Total tasks: {total}")
    logger.info(f"  Successful: {success_count}/{total}")
    logger.info(f"  Total reward: {total_reward:.2f}")
    logger.info(f"  Accuracy: {success_count / total * 100:.1f}%" if total > 0 else "  Accuracy: N/A")

    logger.info("\n  By Level:")
    for lvl, stats in sorted(level_stats.items()):
        acc = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
        logger.info(f"    {lvl}: {stats['success']}/{stats['total']} ({acc:.1f}%)")

    logger.info("\n  By Task Type:")
    for ttype, stats in sorted(type_stats.items()):
        acc = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
        logger.info(f"    {ttype}: {stats['success']}/{stats['total']} ({acc:.1f}%)")

    logger.info(f"\n  Results: {csv_path}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
