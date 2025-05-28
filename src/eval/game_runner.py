import ast
import configparser
import json
import os
import random
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from loguru import logger

from src.eval.guess_game import GuessGame
from src.utils.constants import IND_DOCS_CONFIG
from src.utils.file_manager import FileManager
from src.utils.project_structure import (
    ensure_project_structure,
    get_overall_path,
    get_testset_path,
)


class GameSimulation:
    def __init__(
        self,
        tester_model_name: str,
        testee_model_name: str,
        prompt_strategy: str,
        topic: str,
        verbose: bool,
        num_cards: int = 30,
        random_seed: int = 42,
    ):
        """
        Initialize the game simulation with the given parameters.

        Args:
            tester_model_name: The name of the model asking questions
            testee_model_name: The name of the model answering questions
            prompt_strategy: The strategy for prompting the models
            topic: The topic of the game
            verbose: Whether to print verbose output
            num_cards: Number of cards to use in the simulation
            random_seed: Random seed for reproducibility
        """
        self.tester_model_name = tester_model_name
        self.testee_model_name = testee_model_name
        self.verbose = verbose
        self.topic = topic
        self.num_cards = num_cards
        self.random_seed = random_seed
        self.prompt_strategy = prompt_strategy

        # Make sure all directories exist
        ensure_project_structure()

        # Load configurations
        self.load_ind_docs_config(config_file=IND_DOCS_CONFIG)

        # Get pack path from config
        self.pack_path = self.ind_docs_config.get(self.topic, "pack_path")

        # Load the deck of cards
        self.deck_of_cards = self.load_testset(
            num_cards=num_cards, random_seed=random_seed
        )

    def load_llms_config(self, config_file: str) -> None:
        """
        Load LLM configuration from the given file.

        Args:
            config_file: Path to the configuration file

        Raises:
            FileNotFoundError: If the configuration file is not found
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
        self.llms_config = configparser.ConfigParser()
        self.llms_config.read(config_file)
        self.common_params = {
            "max_tokens": self.llms_config.getint("common_params", "max_tokens"),
            "temperature": self.llms_config.getfloat("common_params", "temperature"),
            # Other common parameters can be added here
        }

    def load_ind_docs_config(self, config_file: str) -> None:
        """
        Load industrial documents configuration from the given file.

        Args:
            config_file: Path to the configuration file

        Raises:
            FileNotFoundError: If the configuration file is not found
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
        self.ind_docs_config = configparser.ConfigParser()
        self.ind_docs_config.read(config_file)

    def load_testset(self, num_cards: int, random_seed: int = 42) -> List[str]:
        """
        Load a deck of cards from a file and perform proportional sampling.

        Args:
            num_cards: Number of cards to select
            random_seed: Random seed for reproducibility

        Returns:
            List of selected cards

        Raises:
            Exception: If there's an error loading the deck
        """
        try:
            random.seed(random_seed)
            testset_path = get_testset_path(self.topic, num_cards)
            selected_cards = []

            # If testset already exists, load it
            if os.path.exists(testset_path):
                logger.info(f"Loading existing testset from {testset_path}")
                selected_cards = ast.literal_eval(FileManager.load_text(testset_path))
            else:
                # If testset doesn't exist, create it
                logger.info(
                    f"Creating new testset for {self.topic} with {num_cards} cards"
                )

                data = None

                if os.path.exists(self.pack_path):
                    logger.info(f"Found card pack at {self.pack_path}")
                    try:
                        with open(self.pack_path, "r") as file:
                            data = json.load(file)["deck"]
                    except Exception as e:
                        logger.warning(f"Failed to load from {self.pack_path}: {e}")

                if data is None:
                    # If no card pack exists, create a sample one
                    logger.warning(
                        f"No card pack found for {self.topic}. Creating a sample pack."
                    )
                    sample_cards = [f"{self.topic}_item_{i}" for i in range(1, 50)]

                    # Save the sample pack for future use
                    sample_data = {"deck": {"final_deck": sample_cards}}
                    os.makedirs(os.path.dirname(self.pack_path), exist_ok=True)
                    with open(self.pack_path, "w") as file:
                        json.dump(sample_data, file, indent=4)

                    data = sample_data["deck"]

                logger.info(f"Deck pack successfully loaded from {self.pack_path}")

                # Get the deck pack
                deck_pack = data.get("cluster_deck", None)

                from src.build.keyword_filter import recall_chain_keywords

                if deck_pack is None or num_cards < len(deck_pack):
                    # No clusters or not enough cards - use the final deck
                    deck_pack = data["final_deck"]
                    starting_card = random.choice(deck_pack)
                    selected_cards = recall_chain_keywords(
                        starting_kw=starting_card,
                        keywords=deck_pack,
                        max_depth=num_cards - 1,
                    )
                else:
                    # Select proportional number of cards from each cluster
                    num_cards_to_select = num_cards // len(deck_pack)
                    for _, cards in deck_pack.items():
                        num_cards_to_select = min(num_cards_to_select, len(cards))
                        starting_card = random.choice(cards)
                        selected_cards.extend(
                            recall_chain_keywords(
                                starting_kw=starting_card,
                                keywords=cards,
                                max_depth=num_cards_to_select - 1,
                            )
                        )

                    # Select the remaining cards randomly
                    remaining_cards = []
                    num_remaining_cards = num_cards - len(selected_cards)

                    if num_remaining_cards > 0:
                        remaining_cards = random.sample(
                            [
                                card
                                for card in data["final_deck"]
                                if card not in selected_cards
                            ],
                            num_remaining_cards,
                        )
                        selected_cards.extend(remaining_cards)

                # Save the selected cards
                os.makedirs(os.path.dirname(testset_path), exist_ok=True)
                FileManager.save_text(selected_cards, str(testset_path))

            # Shuffle the selected cards
            random.shuffle(selected_cards)
            return selected_cards

        except Exception as e:
            logger.error(f"Error loading deck: {str(e)}")
            raise e

    def load_checkpoint(self, deck_of_cards: list):
        run_logs_dir = "outputs/run_logs"
        logger.info(f"Loading checkpoint from {run_logs_dir}")
        # Get the list of run logs
        run_logs = [
            f
            for f in os.listdir(run_logs_dir)
            if (
                f.endswith(".json")
                and f"{self.topic}_{self.tester_model_name}_vs_{self.testee_model_name}_{self.prompt_strategy}_{len(self.deck_of_cards)}"
                in f
            )
        ]
        # Load the run logs and extract the remaining cards
        completed_cards = []
        results = []
        for log in run_logs:
            with open(os.path.join(run_logs_dir, log), "r") as f:
                data = json.load(f)

                if data["chosen_card"] in completed_cards:
                    continue
                completed_cards.append(data["chosen_card"])
                results.append(
                    {
                        "iteration": data["iteration"],
                        "guess_status": data["guess_status"],
                        "chosen_card": data["chosen_card"],
                        "affirmative_rate": data["affirmative_rate"],
                        "invalid_rate": data["invalid_rate"],
                    }
                )

        remaining_cards = [
            card for card in deck_of_cards if card not in completed_cards
        ]
        logger.info(
            f"Loaded {len(completed_cards)} completed cards, {len(remaining_cards)} remaining cards."
        )

        return results, remaining_cards

    def info(self):
        return {
            "tester_model_name": self.tester_model_name,
            "testee_model_name": self.testee_model_name,
            "topic": self.topic,
            "prompt_strategy": self.prompt_strategy,
            "card_count": len(self.deck_of_cards),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "deck_of_cards": self.deck_of_cards,
        }

    def save_results(self, overall: dict):
        """Save the simulation output to a JSON file."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = get_overall_path(
            self.topic,
            self.num_cards,
            self.testee_model_name,
            self.prompt_strategy,
            timestamp,
        )

        FileManager.save_json(overall, str(output_path))
        logger.info(f"Final logs saved to {output_path}")

    def batch_simulation(self, deck_of_cards: list):
        """Run a batch simulation for multiple cards."""
        # Load the checkpoint
        results, remaining_cards = self.load_checkpoint(deck_of_cards)

        # Run the simulation for the remaining cards
        for card in remaining_cards:
            res = self.simulate_round(card)
            results.append(res)
        return results

    def simulate_round(self, chosen_card: str):
        """Simulate a single round of guess game."""
        # Run the guess game simulation
        iteration, guess_status, affirmative_rate, invalid_rate = GuessGame(
            tester_model_name=self.tester_model_name,
            testee_model_name=self.testee_model_name,
            topic=self.topic,
            deck_of_cards=self.deck_of_cards,
            chosen_card=chosen_card,
            max_iterations=self.num_cards,
            prompt_strategy=self.prompt_strategy,
            verbose=self.verbose,
        ).run()

        # Log the results
        logger.info(
            f"Simulation completed for card: {chosen_card}, status: {guess_status}"
        )

        # Log the results
        res = {
            "iteration": iteration,
            "guess_status": guess_status,
            "chosen_card": chosen_card,
            "affirmative_rate": affirmative_rate,
            "invalid_rate": invalid_rate,
        }
        return res

    @staticmethod
    def cal_score(p_model, t_model, t_rand):
        """Calculate the comprehensive score for model evaluation, with penalty based on reasoning steps."""

        # Effectiveness component based on accuracy
        effectiveness = p_model

        # Efficiency component based on iterations using sigmoid to limit penalty in [0,1]

        efficiency = 1 / (1 + np.exp(4 * (t_model - t_rand) / t_rand))

        # Knowledge utilization component with penalty if model takes more steps than random
        knowledge = np.exp(-max(0, (t_model - t_rand) / t_rand))

        # Equal weights for each component
        w1, w2, w3 = 1 / 3, 1 / 3, 1 / 3  # equal weights

        # Combine components to compute the score
        score = w1 * effectiveness + w2 * efficiency + w3 * knowledge

        return score, effectiveness, efficiency, knowledge

    def run(self):
        """Run the simulation."""
        # Run the batch simulation
        logs = self.batch_simulation(self.deck_of_cards)

        # Convert logs into a pandas DataFrame
        df_logs = pd.DataFrame(logs)

        score, e, f, k = self.cal_score(
            p_model=(df_logs["guess_status"] == "correct").mean(),
            t_model=df_logs[df_logs["guess_status"] == "correct"]["iteration"].mean(),
            t_rand=(self.num_cards + 1) / 2,
        )

        # Calculate the overall statistics
        overall = {
            "random_avg_iter_correct": (self.num_cards + 1) / 2,
            "avg_iter": df_logs["iteration"].mean(),
            "avg_affirm": df_logs["affirmative_rate"].mean(),
            "avg_invalid": df_logs["invalid_rate"].mean(),
            "correct_rate": (df_logs["guess_status"] == "correct").mean(),
            "avg_iter_correct": df_logs[df_logs["guess_status"] == "correct"][
                "iteration"
            ].mean(),
            "effectiveness": e,
            "efficiency": f,
            "knowledge_util": k,
            "score": score,
        }

        # Save the output to a JSON file
        self.save_results(
            overall={"info": self.info(), "overall": overall, "logs": logs}
        )
