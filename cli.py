import argparse
import configparser
import os
import shutil
from pathlib import Path

import yaml
from loguru import logger

from src.utils.project_structure import DIRECTORIES, ensure_project_structure

MODELS_CONFIG = "./configs/models.ini"


def parse_args():
    parser = argparse.ArgumentParser(description="Run a game simulation.")

    # Add config file option to the main parser
    parser.add_argument("--config", type=str, help="Path to YAML configuration file.")

    sub_parsers = parser.add_subparsers(
        help="Choose the operation to run.", dest="operation_name"
    )
    # ----------------- Build the deck -----------------
    deck_parser = sub_parsers.add_parser(
        "build_deck", help="Build a keyword deck from a document."
    )
    deck_parser.add_argument(
        "--gen_model", type=str, help="Name of the generation model.", required=True
    )
    deck_parser.add_argument(
        "--topic", type=str, help="Topic for the deck.", required=True
    )
    deck_parser.add_argument(
        "--gen_max_keywords_per_doc",
        type=int,
        help="Maximum number of keywords to generate per document.",
        required=True,
    )

    # ----------------- Run the eval -----------------
    eval_parser = sub_parsers.add_parser("eval", help="Run the evaluation.")
    eval_parser.add_argument(
        "--tester_model",
        type=str,
        help="Name of the tester model. Only one tester is allowed.",
        required=True,
    )
    eval_parser.add_argument(
        "--testee_model",
        type=str,
        nargs="*",
        help="Name of the testee model(s). Use 'all' to evaluate all models in config.ini.",
    )
    eval_parser.add_argument(
        "--topic", type=str, help="Topic for the deck.", required=True
    )
    eval_parser.add_argument(
        "--prompt_strategy", type=str, default="basic", help="Prompt strategy to use."
    )
    eval_parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    eval_parser.add_argument(
        "--num_cards", type=int, default=10, help="Number of cards to simulate."
    )
    eval_parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for the simulation."
    )

    # ----------------- Run the stats -----------------
    stats_parser = sub_parsers.add_parser(
        "stats", help="Generate stats from the results of the game simulations."
    )
    stats_parser.add_argument(
        "--results_dir",
        type=str,
        default="./outputs/overall",
        help="Directory containing the results of the game simulations.",
    )
    stats_parser.add_argument(
        "--stats_dir",
        type=str,
        default="./outputs/stats",
        help="Directory to save the stats.",
    )

    return parser.parse_args()


def read_yaml_config(config_path):
    """Read and parse a YAML configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def run_from_config(config):
    """Run operations based on unified YAML configuration."""
    # Process build_deck operation if enabled
    if config.get("build_deck", {}).get("enabled", False):
        logger.info("Running build_deck operation")
        from src.build.deck_builder import DeckBuilder

        params = config.get("build_deck", {}).get("params", {})
        deck_builder = DeckBuilder(
            gen_model=params.get("gen_model"),
            topic=params.get("topic"),
            gen_max_kw_per_doc=params.get("gen_max_keywords_per_doc"),
        ).run()

    # Process eval operation if enabled
    if config.get("eval", {}).get("enabled", False):
        logger.info("Running eval operation")
        from src.eval.game_runner import GameSimulation

        params = config.get("eval", {}).get("params", {})
        testee_models = params.get("testee_model", [])
        if testee_models and "all" in testee_models:
            config_parser = configparser.ConfigParser()
            config_parser.read(MODELS_CONFIG)
            testee_models = [
                section
                for section in config_parser.sections()
                if section != "common_params"
            ]

        for testee_model in testee_models:
            simulation = GameSimulation(
                tester_model_name=params.get("tester_model"),
                testee_model_name=testee_model,
                topic=params.get("topic"),
                prompt_strategy=params.get("prompt_strategy", "basic"),
                verbose=params.get("verbose", False),
                num_cards=params.get("num_cards", 10),
                random_seed=params.get("random_seed", 42),
            )
            simulation.run()

    # Process stats operation if enabled
    if config.get("stats", {}).get("enabled", False):
        logger.info("Running stats operation")
        from src.utils.stats import stats

        params = config.get("stats", {}).get("params", {})
        stats(
            results_dir=params.get("results_dir", "./outputs/overall"),
            stats_dir=params.get("stats_dir", "./outputs/stats"),
        )


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Start CLI with args: {args}")

    migrate_legacy_files()

    if args.config:
        # Run operations from unified YAML configuration
        config = read_yaml_config(args.config)
        run_from_config(config)
    elif args.operation_name == "build_deck":
        from src.build.deck_builder import DeckBuilder

        deck_builder = DeckBuilder(
            gen_model=args.gen_model,
            topic=args.topic,
            gen_max_kw_per_doc=args.gen_max_keywords_per_doc,
        ).run()
    elif args.operation_name == "eval":
        from src.eval.game_runner import GameSimulation

        if args.testee_model and "all" in args.testee_model:
            config_parser = configparser.ConfigParser()
            config_parser.read(MODELS_CONFIG)
            testee_models = [
                section
                for section in config_parser.sections()
                if section != "common_params"
            ]
        else:
            testee_models = args.testee_model

        for testee_model in testee_models:
            simulation = GameSimulation(
                tester_model_name=args.tester_model,
                testee_model_name=testee_model,
                topic=args.topic,
                prompt_strategy=args.prompt_strategy,
                verbose=args.verbose,
                num_cards=args.num_cards,
                random_seed=args.random_seed,
            )
            simulation.run()

    elif args.operation_name == "stats":
        from src.utils.stats import stats

        stats(results_dir=args.results_dir, stats_dir=args.stats_dir)
    else:
        raise ValueError("Invalid operation name.")
