"""
OmniGAIA Benchmark with Orchestra-o1 (MainAgent + SubAgent)

OmniGAIA is a multimodal benchmark that includes video, audio, and image inputs.
Key differences from GAIA:
- Video analysis support (VideoAnalysisAction)
- Multimodal input list (omni_modal_input)
- Difficulty levels are strings (Easy/Medium/Hard)

Usage:
    python bench_orchestra_o1_omnigaia.py --config config/benchmarks/orchestra_o1_omnigaia.yaml
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.engine.logs import logger
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


DEFAULT_CONFIG_PATH = ROOT / "config/benchmarks/orchestra_o1_omnigaia.yaml"


async def main():
    """Run OmniGAIA benchmark with Orchestra-o1."""
    parser = argparse.ArgumentParser(description="Run OmniGAIA benchmark using Orchestra-o1.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML.")
    parser.add_argument("--max_concurrency", type=int, default=None, help="Override max_concurrency.")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated task IDs.")
    parser.add_argument("--skip_completed", type=str, default=None, help="Path to existing CSV to skip completed tasks.")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("OmniGAIA Benchmark with Orchestra-o1")
    logger.info("=" * 60)

    # Load configuration (reuse GAIAOrchestraConfig to load common YAML fields)
    cfg = GAIAOrchestraConfig.load(args.config)

    # Check dataset
    if not cfg.dataset_path.exists():
        logger.error(f"Dataset not found: {cfg.dataset_path}")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.timestamp = timestamp

    logger.info(f"Dataset: {cfg.dataset_path}")
    logger.info(f"Attachments: {cfg.attachments_dir}")
    logger.info(f"Level filter: {cfg.level_filter}")

    # Create OmniGAIA tools (adds VideoAnalysisAction compared to GAIA)
    omnigaia_tools = [
        GoogleSearchAction(),
        ExecuteCodeAction(),
        ExtractUrlContentAction(),
        ImageAnalysisAction(),
        ParseAudioAction(),
        VideoAnalysisAction(),
    ]
    logger.info(f"Loaded {len(omnigaia_tools)} OmniGAIA tools: {[t.name for t in omnigaia_tools]}")

    # Create OmniGAIA config and benchmark
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

    # Prepare output paths
    cfg.result_folder.mkdir(parents=True, exist_ok=True)
    cfg.trajectory_folder.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.result_folder / f"omnigaia_orchestra_o1_{timestamp}.csv"

    # Create runner
    runner = OmniGAIARunner(
        benchmark=benchmark,
        main_model=cfg.main_model,
        sub_models=cfg.sub_models,
        max_attempts=cfg.max_attempts,
        omnigaia_tools=omnigaia_tools,
    )

    logger.info(f"[OmniGAIA] main_model={cfg.main_model}, sub_models={cfg.sub_models}")
    logger.info(f"Max concurrency: {args.max_concurrency or cfg.max_concurrency}")
    logger.info(f"Results: {csv_path}")

    # Run benchmark
    results = await runner.run_levels(
        levels=levels,
        max_concurrency=args.max_concurrency or cfg.max_concurrency,
        csv_path=csv_path,
        trajectory_folder=cfg.trajectory_folder,
        timestamp=timestamp,
    )

    # Summary
    total = len(results)
    success_count = sum(1 for r in results.values() if r.get("success"))
    total_reward = sum(float(r.get("reward", 0) or 0) for r in results.values())

    # Statistics by difficulty level
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

    # Statistics by task type
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
    logger.info("OmniGAIA Benchmark Summary:")
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
