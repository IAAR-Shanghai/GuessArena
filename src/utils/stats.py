import csv
import json
import os
from datetime import datetime

from loguru import logger


def stats(results_dir: str = "./outputs/overall", stats_dir: str = "./outputs/stats"):
    """Generate stats from the results of the game simulations."""
    results_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]

    latest_files = {}
    for f in results_files:
        prefix = "_".join(f.split("_")[:-2])
        timestamp = "_".join(f.split("_")[-2:]).strip(".json")
        latest_files_timestamp = "_".join(
            latest_files.get(prefix, "").split("_")[-2:]
        ).strip(".json")
        if prefix not in latest_files or timestamp > latest_files_timestamp:
            latest_files[prefix] = f

    results_files = list(latest_files.values())

    stats = []
    for file in results_files:
        with open(os.path.join(results_dir, file), "r") as f:
            data = json.load(f)
            info = data["info"]
            overall = data["overall"]
            stats.append(
                {
                    "tester_model": info["tester_model_name"],
                    "testee_model": info["testee_model_name"],
                    "topic": info["topic"],
                    "prompt_strategy": info["prompt_strategy"],
                    "card_count": info["card_count"],
                    "avg_iterations": overall["avg_iter"],
                    "avg_affirmative_rate": overall["avg_affirm"],
                    "avg_invalid_rate": overall["avg_invalid"],
                    "correct_guess_rate": overall["correct_rate"],
                    "avg_iteration_guess_correct": overall["avg_iter_correct"],
                    "score": overall["score"],
                }
            )

    # Create the stats directory if it doesn't exist
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    # Save the stats to a csv file
    csv_file = os.path.join(
        stats_dir, f"stats_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    )
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stats[0].keys())
        writer.writeheader()
        writer.writerows(stats)

    logger.info(f"Stats saved to {csv_file}")
