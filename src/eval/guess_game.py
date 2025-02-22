import configparser
import json
import os
from datetime import datetime

from loguru import logger
from openai import OpenAI

from src.utils.kbg_gen import KnowBackgroundGenerator

# Define paths for system prompts
TESTER_SYSTEM_PATH = "./prompts/tester_system.txt"
BASIC_TESTEE_SYSTEM_PATH = "./prompts/basic_testee_system.txt"
COT_TESTEE_SYSTEM_PATH = "./prompts/cot_testee_system.txt"
KNOW_TESTEE_SYSTEM_PATH = "./prompts/know_testee_system.txt"

MODELS_CONFIG = "./configs/models.ini"


class GuessGame:
    def __init__(
        self,
        tester_model_name: str,
        testee_model_name: str,
        topic: str,
        deck_of_cards: list[str],
        chosen_card: str,
        max_iterations: int = 30,
        prompt_strategy: str = "basic",
        verbose: bool = False,
    ):
        self.load_llms_config(config_file=MODELS_CONFIG)

        self.tester_model_name = tester_model_name
        self.testee_model_name = testee_model_name

        self.tester_model = OpenAI(
            base_url=self.llms_config.get(self.tester_model_name, "base_url"),
            api_key=self.llms_config.get(self.tester_model_name, "api_key"),
        )
        self.testee_model = OpenAI(
            base_url=self.llms_config.get(self.testee_model_name, "base_url"),
            api_key=self.llms_config.get(self.testee_model_name, "api_key"),
        )

        self.topic = topic
        self.deck_of_cards = deck_of_cards
        self.chosen_card = chosen_card
        self.max_iterations = max_iterations
        self.verbose = verbose

        self.conversation_log = []

        self.tester_system_prompt = self.load_prompt(TESTER_SYSTEM_PATH).format(
            deck_of_cards=self.deck_of_cards, chosen_card=self.chosen_card
        )

        self.tester_message_log = [
            {"role": "system", "content": self.tester_system_prompt}
        ]

        self.prompt_strategy = prompt_strategy
        if self.prompt_strategy == "basic":
            self.testee_system_prompt = self.load_prompt(
                BASIC_TESTEE_SYSTEM_PATH
            ).format(deck_of_cards=self.deck_of_cards)
        elif self.prompt_strategy == "cot":
            self.testee_system_prompt = self.load_prompt(COT_TESTEE_SYSTEM_PATH).format(
                deck_of_cards=self.deck_of_cards
            )
        elif self.prompt_strategy == "know":
            kbg = KnowBackgroundGenerator(
                gen_model="GPT_4o", topic=self.topic, deck_of_cards=self.deck_of_cards
            )
            kbg = kbg.run()
            self.testee_system_prompt = self.load_prompt(
                KNOW_TESTEE_SYSTEM_PATH
            ).format(deck_of_cards=self.deck_of_cards, knowledge_background=kbg)

        self.testee_message_log = [
            {"role": "system", "content": self.testee_system_prompt}
        ]

    @staticmethod
    def load_prompt(file_path: str):
        """Load the contents of a prompt text file."""
        try:
            with open(file_path, "r") as file:
                return file.read()
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(
                f"Critical system prompt file is missing: {file_path}"
            )

    def load_llms_config(self, config_file: str):
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
        self.llms_config = configparser.ConfigParser()
        self.llms_config.read(config_file)
        self.common_params = {
            "max_tokens": self.llms_config.getint("common_params", "max_tokens"),
            "temperature": self.llms_config.getfloat("common_params", "temperature"),
            # Other common parameters can be added here
        }

    def send_message(self, sender, sender_name, sender_messages, receiver_messages):
        """Send a message from one model to another."""
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

    def save_log(self, results: dict, output_dir: str = "outputs/run_logs"):
        """Save the simulation log to a JSON file."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Replace special characters in the chosen card name
        chosen_card = (
            self.chosen_card.replace(" ", "_").replace("/", "_").replace("\\", "_")
        )
        filename = f"{self.topic}_{self.tester_model_name}_vs_{self.testee_model_name}_{self.prompt_strategy}_{len(self.deck_of_cards)}_{chosen_card}_{results['timestamp']}.json"
        output_path = os.path.join(output_dir, filename)

        with open(output_path, "w") as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

        logger.info(f"Final logs saved to {output_path}")

    def run(self):
        """Play the guessing game simulation."""
        iteration = 0
        guess_status = "incorrect"
        status_code = -1
        affirmative_responses = 0
        invalid_responses = 0

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
                if self.chosen_card.strip().lower() in testee_question.strip().lower():
                    guess_status = "correct"
                break

        affirmative_rate = round(affirmative_responses / iteration, 6)
        invalid_rate = round(invalid_responses / iteration, 6)

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
