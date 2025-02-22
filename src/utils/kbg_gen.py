import configparser
import json
import os

from langchain_openai import ChatOpenAI
from loguru import logger

PROMPT_PATH = "./prompts/gen_kbg.txt"
MODELS_CONFIG = "./configs/models.ini"
IND_DOCS_CONFIG = "./configs/ind_docs.ini"


class KnowBackgroundGenerator:
    def __init__(self, gen_model: str, topic: str, deck_of_cards: list[str]):
        self.gen_model = gen_model
        self.topic = topic
        self.deck_of_cards = deck_of_cards

        self.load_llms_config(config_file=MODELS_CONFIG)
        self.load_ind_docs_config(config_file=IND_DOCS_CONFIG)
        self.prompt_template = self.load_prompt(PROMPT_PATH)

    def load_llms_config(self, config_file: str):
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
        self.llms_config = configparser.ConfigParser()
        self.llms_config.read(config_file)
        self.common_params = {
            "max_tokens": self.llms_config.getint(
                "common_params", "max_tokens", fallback=2048
            ),
            "temperature": self.llms_config.getfloat(
                "common_params", "temperature", fallback=0.7
            ),
            # Other common parameters can be added here
        }

    def load_ind_docs_config(self, config_file: str):
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
        self.ind_docs_config = configparser.ConfigParser()
        self.ind_docs_config.read(config_file)

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

    def initialize_model(self, model_name: str):
        try:
            return ChatOpenAI(
                model=self.llms_config.get(
                    model_name, "model", fallback="default_model"
                ),
                base_url=self.llms_config.get(
                    model_name, "base_url", fallback="default_base_url"
                ),
                api_key=self.llms_config.get(
                    model_name, "api_key", fallback="default_api_key"
                ),
                **self.common_params,
            )
        except KeyError as e:
            raise ValueError(f"Missing configuration for {model_name}: {e}")

    def save_kbg(self, kbg: str):
        """Save the knowledge background to a text file."""
        with open(f"outputs/kbg-files/{self.topic}_kbg.txt", "w") as file:
            file.write(kbg)

    def gen_kbg(self):
        """Generate a knowledge background text."""
        logger.info(f"Generating knowledge background for topic '{self.topic}'...")
        model = self.initialize_model(self.gen_model)
        prompt = self.prompt_template.format(
            name=self.ind_docs_config.get(self.topic, "name"),
            description=self.ind_docs_config.get(self.topic, "description"),
            deck_of_cards=self.deck_of_cards,
        )
        response = (
            model.invoke(prompt)
            .content.replace("```json", "")
            .replace("```", "")
            .strip()
        )
        response = json.loads(response).get("knowledge_background")
        self.save_kbg(response)
        return response

    def run(self):
        """Run the knowledge background generator."""

        # Check if the KBG file already exists
        if not os.path.exists(f"outputs/kbg-files/{self.topic}_kbg.txt"):
            self.gen_kbg()

        # Load the generated KBG file
        with open(f"outputs/kbg-files/{self.topic}_kbg.txt", "r") as f:
            kbg = f.read()
        logger.info(f"Lodaed knowledge background for topic '{self.topic}' ...")

        return kbg


if __name__ == "__main__":
    kbg = KnowBackgroundGenerator(
        gen_model="GPT_4o",
        topic="info_tech",
        deck_of_cards=[
            "secure access service edge",
            "quantum computing transition",
            "ai model cost",
            "iot market pressures",
            "check point",
            "it data integration",
            "iot connection strategy",
            "compute power rise",
            "ai regulation",
            "home automation",
        ],
    )
    kbg.gen_kbg()
