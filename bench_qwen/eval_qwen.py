"""
Evaluation result summary script - Qwen3-8B MainAgent version
Read Qwen3-8B evaluation results from CSV and generate summary report.

Usage:
    # Auto-find latest CSV file
    python bench_qwen/eval_qwen.py
    
    # Specify CSV file
    python bench_qwen/eval_qwen.py --csv_path logs/omnigaia_qwen/omnigaia_qwen_xxx.csv
"""

import csv
import os
import argparse
import glob
from datetime import datetime
from collections import defaultdict

TOTAL_SAMPLES = 360
DEFAULT_MAIN_AGENT = "qwen3-8b"


def find_latest_csv(output_dir):
    """Find the latest CSV file."""
    pattern = os.path.join(output_dir, "omnigaia_qwen_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def parse_datetime(dt_str):
    """Parse datetime string."""
    if not dt_str:
        return None
    dt_str = dt_str.strip()
    try:
        return datetime.fromisoformat(dt_str)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description="OmniGAIA Qwen3-8B Evaluation result summary")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="CSV file path (auto-finds latest if not specified)")
    parser.add_argument("--main_agent", type=str, default=DEFAULT_MAIN_AGENT,
                        help="Main model name, default qwen3-8b")
    args = parser.parse_args()
    main_agent = args.main_agent

    # Determine CSV file path
    if args.csv_path:
        csv_path = args.csv_path
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        output_dir = os.path.join(project_dir, "logs", "omnigaia_qwen")
        csv_path = find_latest_csv(output_dir)
        if not csv_path:
            print(f"Error: no CSV file found")
            print("Please run evaluation first: python bench_qwen/bench_qwen_omnigaia.py")
            return

    output_dir = os.path.dirname(csv_path)
    output_path = os.path.join(output_dir, f"results_{main_agent}.txt")

    print(f"Reading CSV: {csv_path}")

    # # Read CSV data
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("Error: CSV file is empty")
        return

    # ========== 1. Overall statistics ==========
    total_success = sum(1 for r in rows if r["success"].strip() == "True")
    total_cost = sum(float(r["total_cost"]) for r in rows)
    overall_accuracy = total_success / TOTAL_SAMPLES

    # ========== 2. Statistics by category ==========
    category_stats = defaultdict(lambda: {"success": 0, "total": 0, "cost": 0.0})
    for r in rows:
        cat = r["category"].strip()
        category_stats[cat]["total"] += 1
        category_stats[cat]["cost"] += float(r["total_cost"])
        if r["success"].strip() == "True":
            category_stats[cat]["success"] += 1

    # ========== 3. Statistics by level ==========
    level_stats = defaultdict(lambda: {"success": 0, "total": 0, "cost": 0.0})
    for r in rows:
        lvl = r["level"].strip()
        level_stats[lvl]["total"] += 1
        level_stats[lvl]["cost"] += float(r["total_cost"])
        if r["success"].strip() == "True":
            level_stats[lvl]["success"] += 1

    # ========== 4. Calculate total elapsed time ==========
    all_start_times = []
    all_end_times = []
    for r in rows:
        st = parse_datetime(r.get("start_time", ""))
        et = parse_datetime(r.get("end_time", ""))
        if st:
            all_start_times.append(st)
        if et:
            all_end_times.append(et)

    if all_start_times and all_end_times:
        earliest_start = min(all_start_times)
        latest_end = max(all_end_times)
        total_elapsed = latest_end - earliest_start
        total_seconds = int(total_elapsed.total_seconds())
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        elapsed_readable = f"{days}d {hours}h {minutes}m {seconds}s"
        elapsed_seconds = f"{total_seconds}s"
    else:
        elapsed_readable = "N/A"
        elapsed_seconds = "N/A"

    # ========== # Generate report ==========
    lines = []
    lines.append("=" * 60)
    lines.append(f"  OmniGAIA Orchestra-o1 Evaluation result summary /  Main Agent: {main_agent}")
    lines.append("=" * 60)

    # # Overall
    lines.append("【1. Overall statistics】")
    lines.append(f"  Total samples:  {TOTAL_SAMPLES}")
    lines.append(f"  Successful:     {total_success}")
    lines.append(f"  # Overall Accuracy:  {overall_accuracy * 100:.1f}% ({total_success}/{TOTAL_SAMPLES})")
    lines.append(f"  # Overall Total Cost: ${total_cost:.1f}")
    lines.append("")

    # # By Category
    lines.append("【2. Statistics by category】")
    lines.append(f"  {'Category':<30s} {'Accuracy':>10s} {'Success':>8s} {'Total':>6s} {'Cost':>12s}")
    lines.append("  " + "-" * 70)
    for cat in sorted(category_stats.keys()):
        s = category_stats[cat]
        acc = s["success"] / s["total"] * 100 if s["total"] > 0 else 0.0
        cost_str = f"${s['cost']:.1f}"
        lines.append(f"  {cat:<30s} {acc:>9.1f}% {s['success']:>8d} {s['total']:>6d} {cost_str:>12s}")
    lines.append("")

    # # By Level
    lines.append("[3. By Level (Difficulty)]")
    lines.append(f"  {'Level':<15s} {'Accuracy':>10s} {'Success':>8s} {'Total':>6s} {'Cost':>12s}")
    lines.append("  " + "-" * 55)
    for lvl in ["Easy", "Medium", "Hard"]:
        if lvl in level_stats:
            s = level_stats[lvl]
            acc = s["success"] / s["total"] * 100 if s["total"] > 0 else 0.0
            cost_str = f"${s['cost']:.1f}"
            lines.append(f"  {lvl:<15s} {acc:>9.1f}% {s['success']:>8d} {s['total']:>6d} {cost_str:>12s}")
    lines.append("")

    # # Total elapsed time
    lines.append("【4. # Total elapsed time】")
    if all_start_times and all_end_times:
        lines.append(f"  Earliest start: {earliest_start.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Latest end:     {latest_end.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Total elapsed:  {elapsed_readable}")
        lines.append(f"  Total elapsed:  {elapsed_seconds}")
    else:
        lines.append(f"  Total elapsed: N/A")
    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)

    # # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    # # Also print to console
    print(report)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
