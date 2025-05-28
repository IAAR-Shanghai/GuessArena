import json
import os
from datetime import datetime
from typing import Any, List

from loguru import logger


class FileManager:
    """Utility class to manage file operations across the project."""

    @staticmethod
    def ensure_directories_exist(directories: List[str]) -> None:
        """
        Ensure all the specified directories exist.

        Args:
            directories: List of directory paths to create
        """
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def save_json(data: Any, filepath: str, ensure_dir: bool = True) -> None:
        """
        Save data to a JSON file.

        Args:
            data: Data to save
            filepath: Path to the output file
            ensure_dir: Whether to ensure the directory exists
        """
        if ensure_dir:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"Data saved to {filepath}")

    @staticmethod
    def save_text(data: str, filepath: str, ensure_dir: bool = True) -> None:
        """
        Save text to a file.

        Args:
            data: Text to save
            filepath: Path to the output file
            ensure_dir: Whether to ensure the directory exists
        """
        if ensure_dir:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"Text saved to {filepath}")

    @staticmethod
    def load_json(filepath: str) -> Any:
        """
        Load data from a JSON file.

        Args:
            filepath: Path to the input file

        Returns:
            Loaded data

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File '{filepath}' not found.")

        with open(filepath, "r") as f:
            data = json.load(f)
        return data

    @staticmethod
    def load_text(filepath: str) -> str:
        """
        Load text from a file.

        Args:
            filepath: Path to the text file

        Returns:
            Content of the file as a string

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File '{filepath}' not found.")

        with open(filepath, "r") as f:
            return f.read()

    @staticmethod
    def get_output_path(
        base_dir: str,
        topic: str,
        num_cards: int,
        model_name: str,
        prompt_strategy: str,
        extension: str = "json",
    ) -> str:
        """
        Generate a standardized output path.

        Args:
            base_dir: Base directory for outputs
            topic: Topic of the simulation
            num_cards: Number of cards used
            model_name: Name of the model
            prompt_strategy: Prompt strategy used
            extension: File extension

        Returns:
            Generated output path
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{topic}_{num_cards}_{model_name}_{prompt_strategy}_{timestamp}.{extension}"
        return os.path.join(base_dir, filename)
