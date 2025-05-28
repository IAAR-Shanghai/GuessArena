import configparser
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

from loguru import logger
from openai import OpenAI

from src.utils.constants import (
    BASIC_TESTEE_SYSTEM_PATH,
    COT_TESTEE_SYSTEM_PATH,
    KNOW_TESTEE_SYSTEM_PATH,
    MODELS_CONFIG,
    TESTER_SYSTEM_PATH,
)
from src.utils.file_manager import FileManager
from src.utils.kbg_gen import KnowBackgroundGenerator
from src.utils.project_structure import (
    ensure_project_structure,
    get_knowledge_path,
    get_run_log_path,
)


class GuessGame:
    def __init__(
        self,
        tester_model_name: str,
        testee_model_name: str,
        topic: str,
        deck_of_cards: List[str],
        chosen_card: str,
        max_iterations: int = 30,
        prompt_strategy: str = "basic",
        verbose: bool = False,
    ):
        """
        Initialize the GuessGame with the given parameters.

        Args:
            tester_model_name: Name of the model that asks questions
            testee_model_name: Name of the model that guesses
            topic: Topic of the game
            deck_of_cards: List of available cards
            chosen_card: Card to be guessed
            max_iterations: Maximum number of iterations allowed
            prompt_strategy: Strategy for prompting (basic, cot, know)
            verbose: Whether to print verbose output
        """
        # Ensure all directories exist
        ensure_project_structure()

        # Load LLM configuration
        self.load_llms_config(config_file=MODELS_CONFIG)

        self.tester_model_name = tester_model_name
        self.testee_model_name = testee_model_name

        # Initialize model clients
        try:
            self.tester_model = OpenAI(
                base_url=self.llms_config.get(self.tester_model_name, "base_url"),
                api_key=self.llms_config.get(self.tester_model_name, "api_key"),
            )
            self.testee_model = OpenAI(
                base_url=self.llms_config.get(self.testee_model_name, "base_url"),
                api_key=self.llms_config.get(self.testee_model_name, "api_key"),
            )
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise RuntimeError(f"Failed to initialize models: {e}")

        self.topic = topic
        self.deck_of_cards = deck_of_cards
        self.chosen_card = chosen_card
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.prompt_strategy = prompt_strategy

        self.conversation_log = []

        # Load and format system prompts
        try:
            self.tester_system_prompt = FileManager.load_text(
                TESTER_SYSTEM_PATH
            ).format(deck_of_cards=self.deck_of_cards, chosen_card=self.chosen_card)

            self.tester_message_log = [
                {"role": "system", "content": self.tester_system_prompt}
            ]

            # Set up testee system prompt based on the strategy
            self._setup_testee_prompt()

        except Exception as e:
            logger.error(f"Error setting up prompts: {e}")
            raise RuntimeError(f"Failed to set up game prompts: {e}")

    def _setup_testee_prompt(self) -> None:
        """Set up the testee system prompt based on the selected strategy."""
        try:
            if self.prompt_strategy == "basic":
                self.testee_system_prompt = FileManager.load_text(
                    BASIC_TESTEE_SYSTEM_PATH
                ).format(deck_of_cards=self.deck_of_cards)
            elif self.prompt_strategy == "cot":
                self.testee_system_prompt = FileManager.load_text(
                    COT_TESTEE_SYSTEM_PATH
                ).format(deck_of_cards=self.deck_of_cards)
            elif self.prompt_strategy == "know":
                # Check if knowledge background exists
                knowledge_path = get_knowledge_path(self.topic)
                if os.path.exists(knowledge_path):
                    # Load existing knowledge background
                    knowledge_background = FileManager.load_text(knowledge_path)
                else:
                    # Generate new knowledge background
                    kbg = KnowBackgroundGenerator(
                        gen_model="GPT-4o",
                        topic=self.topic,
                        deck_of_cards=self.deck_of_cards,
                    )
                    knowledge_background = kbg.run()
                    # Save the knowledge background for future use
                    FileManager.save_txt(knowledge_background, str(knowledge_path))

                self.testee_system_prompt = FileManager.load_text(
                    KNOW_TESTEE_SYSTEM_PATH
                ).format(
                    deck_of_cards=self.deck_of_cards,
                    knowledge_background=knowledge_background,
                )
            else:
                logger.warning(
                    f"Unknown prompt strategy: {self.prompt_strategy}, falling back to basic"
                )
                self.testee_system_prompt = FileManager.load_text(
                    BASIC_TESTEE_SYSTEM_PATH
                ).format(deck_of_cards=self.deck_of_cards)

            self.testee_message_log = [
                {"role": "system", "content": self.testee_system_prompt}
            ]
        except Exception as e:
            logger.error(f"Error setting up testee prompt: {e}")
            raise RuntimeError(f"Failed to set up testee prompt: {e}")

    @staticmethod
    def load_prompt(file_path: str) -> str:
        """
        Load the contents of a prompt text file.

        Args:
            file_path: Path to the prompt file

        Returns:
            Content of the prompt file

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        return FileManager.load_text(file_path)

    def load_llms_config(self, config_file: str) -> None:
        """
        Load LLM configuration from file.

        Args:
            config_file: Path to the configuration file

        Raises:
            FileNotFoundError: If the file doesn't exist
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

    def send_message(
        self, sender, sender_name, sender_messages, receiver_messages
    ) -> str:
        """
        Send a message from one model to another.

        Args:
            sender: The model sending the message
            sender_name: Name of the sender model
            sender_messages: Message history of the sender
            receiver_messages: Message history of the receiver

        Returns:
            Response from the sender

        Raises:
            Exception: If there's an error during message exchange
        """
        try:
            response_obj = sender.chat.completions.create(
                model=self.llms_config.get(sender_name, "model"),
                messages=sender_messages,
                **self.common_params,
            )
            sender_response = response_obj.choices[0].message.content
            sender_messages.append({"role": "assistant", "content": sender_response})
            receiver_messages.append({"role": "user", "content": sender_response})
            self.conversation_log.append(f"{sender_name}: {sender_response}")
            return sender_response
        except Exception as e:
            logger.error(f"Error during message exchange: {e}")
            raise

    def save_log(self, results: Dict[str, Any]) -> None:
        """
        Save the simulation log to a JSON file.

        Args:
            results: Results to save
        """
        timestamp = results.get(
            "timestamp", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )

        output_path = get_run_log_path(
            self.topic,
            self.tester_model_name,
            self.testee_model_name,
            self.prompt_strategy,
            len(self.deck_of_cards),
            self.chosen_card,
            timestamp,
        )

        FileManager.save_json(results, str(output_path))
        logger.info(f"Run log saved to {output_path}")

    def run(self) -> Tuple[int, str, float, float]:
        """
        Play the guessing game simulation.

        Returns:
            Tuple containing:
            - Number of iterations
            - Guess status ("correct" or "incorrect")
            - Rate of affirmative responses
            - Rate of invalid responses
        """
        iteration = 0
        guess_status = "incorrect"
        status_code = -1
        affirmative_responses = 0
        invalid_responses = 0

        try:
            while iteration < self.max_iterations:
                iteration += 1

                # Testee asks a question, Tester responds
                testee_question = self.send_message(
                    self.testee_model,
                    self.testee_model_name,
                    self.testee_message_log,
                    self.tester_message_log,
                )
                tester_response = self.send_message(
                    self.tester_model,
                    self.tester_model_name,
                    self.tester_message_log,
                    self.testee_message_log,
                )

                if self.verbose:
                    print(f"Testee: {testee_question}")
                    print(f"Tester: {tester_response}")

                if "[Yes]" in tester_response:
                    affirmative_responses += 1

                if "[Invalid]" in tester_response:
                    invalid_responses += 1

                if "[End]" in tester_response:
                    status_code = 0
                    if (
                        self.chosen_card.strip().lower()
                        in testee_question.strip().lower()
                    ):
                        guess_status = "correct"
                    break

            affirmative_rate = round(affirmative_responses / max(1, iteration), 6)
            invalid_rate = round(invalid_responses / max(1, iteration), 6)

            logs = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
                "status_code": status_code,
                "iteration": iteration,
                "max_iter": self.max_iterations,
                "affirmative_rate": affirmative_rate,
                "invalid_rate": invalid_rate,
                "tester": self.tester_model_name,
                "testee": self.testee_model_name,
                "chosen_card": self.chosen_card,
                "guess_status": guess_status,
                "deck_of_cards": self.deck_of_cards,
                "conversation": self.conversation_log,
                "tester_messages": [msg for msg in self.tester_message_log],
                "testee_messages": [msg for msg in self.testee_message_log],
            }

            self.save_log(logs)

            return iteration, guess_status, affirmative_rate, invalid_rate

        except Exception as e:
            logger.error(f"Error during game simulation: {e}")
            # Save partial results if possible
            try:
                affirmative_rate = round(affirmative_responses / max(1, iteration), 6)
                invalid_rate = round(invalid_responses / max(1, iteration), 6)

                logs = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
                    "status_code": -1,  # Error status
                    "iteration": iteration,
                    "max_iter": self.max_iterations,
                    "affirmative_rate": affirmative_rate,
                    "invalid_rate": invalid_rate,
                    "tester": self.tester_model_name,
                    "testee": self.testee_model_name,
                    "chosen_card": self.chosen_card,
                    "guess_status": "error",
                    "error": str(e),
                    "deck_of_cards": self.deck_of_cards,
                    "conversation": self.conversation_log,
                }
                self.save_log(logs)
            except Exception as save_error:
                logger.error(f"Failed to save error log: {save_error}")

            raise RuntimeError(f"Game simulation failed: {e}")
