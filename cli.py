import argparse
import configparser

from loguru import logger

MODELS_CONFIG = "./configs/models.ini"


def parse_args():
    parser = argparse.ArgumentParser(description="Run a game simulation.")
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


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Start CLI with args: {args}")

    if args.operation_name == "build_deck":
        from src.build.deck_builder import DeckBuilder

        # python cli.py build_deck --gen_model GPT_4o --topic info_tech --gen_max_keywords_per_doc 100
        deck_builder = DeckBuilder(
            gen_model=args.gen_model,
            topic=args.topic,
            gen_max_kw_per_doc=args.gen_max_keywords_per_doc,
        ).run()
    elif args.operation_name == "eval":
        from src.eval.game_runner import GameSimulation

        # python cli.py eval --tester_model GPT_4o --testee_model GPT_4o --topic info_tech --prompt_strategy basic --verbose --num_cards 10 --random_seed 42
        if args.testee_model and "all" in args.testee_model:
            config = configparser.ConfigParser()
            config.read(MODELS_CONFIG)
            testee_models = [
                section for section in config.sections() if section != "common_params"
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

        # python cli.py stats
        stats(results_dir=args.results_dir, stats_dir=args.stats_dir)
    else:
        raise ValueError("Invalid operation name.")
