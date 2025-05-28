import configparser
import json
import os
import string
import warnings
from datetime import datetime

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from loguru import logger

from src.build.keyword_filter import cluster_keywords_with_graph, filter_by_sim

warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
UPPER_SIM_THRESHOLD = 0.9
LOWER_SIM_THRESHOLD = 0.35

PROMPT_PATH = "./prompts/gen_kws_deck.txt"
MODELS_CONFIG = "./configs/models.ini"
IND_DOCS_CONFIG = "./configs/ind_docs.ini"


class DeckBuilder:
    def __init__(
        self, gen_model: list, topic: str, gen_max_kw_per_doc: int, num_cards: int = 20
    ):
        self.gen_model = gen_model
        self.topic = topic
        self.gen_max_kw_per_doc = gen_max_kw_per_doc
        self.n_clusters = 10

        self.load_llms_config(config_file=MODELS_CONFIG)
        self.load_ind_docs_config(config_file=IND_DOCS_CONFIG)
        self.docs_dir = self.ind_docs_config.get(self.topic, "docs_dir")
        self.prompt_template = self.load_prompt(PROMPT_PATH)

        self.embedding_model = EMBEDDING_MODEL
        self.upper_sim_threshold = UPPER_SIM_THRESHOLD
        self.lower_sim_threshold = LOWER_SIM_THRESHOLD
        self.num_cards = num_cards

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
        """Create a ChatOpenAI model instance with error handling."""
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

    def gen_init_deck(self, model_name: str, document_path: str):
        """Generate an initial deck of cards from a document."""
        # Load the document
        loader = PyPDFLoader(document_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents from {document_path}")

        # Embed the document
        vector_store = FAISS.from_documents(documents, self.embedding_model)
        retriever = vector_store.as_retriever()
        logger.info("Document embedding complete!")

        # Initialize the language model
        llm = self.initialize_model(model_name)

        # Set prompt
        query = self.prompt_template.format(
            name=self.ind_docs_config.get(self.topic, "name"),
            description=self.ind_docs_config.get(self.topic, "description"),
            max_keywords=self.gen_max_kw_per_doc,
        )

        # Create a RAG chain
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        # Invoke the RAG chain
        result = rag_chain.invoke({"query": query})["result"]
        keywords = [
            kw.strip().lower().strip(string.punctuation)
            for kw in set(result.split(";"))
            if len(kw.strip()) >= 3
        ]
        logger.info(
            f"Successfully generated initial deck of {self.gen_max_kw_per_doc} keywords by {model_name} from {document_path} ..."
        )

        return keywords

    def batch_gen(self, model_name: str):
        """Generate initial decks from multiple documents."""
        deck = []
        docs_lst = [doc for doc in os.listdir(self.docs_dir) if doc.endswith(".pdf")]
        for doc in docs_lst:
            doc_path = os.path.join(self.docs_dir, doc)
            try:
                deck.extend(self.gen_init_deck(model_name, doc_path))
            except Exception as e:
                logger.error(
                    f"Failed to generate initial deck with {model_name} from {doc_path}: {e}"
                )

        return deck

    def filter_and_cluster_keywords(self, init_deck, n_clusters):
        """Filter and cluster keywords."""
        deck = list(set(init_deck))
        final_deck, _ = filter_by_sim(
            topic=self.ind_docs_config.get(self.topic, "name"),
            keywords=deck,
            lower_thresh=self.lower_sim_threshold,
            upper_thresh=self.upper_sim_threshold,
        )
        cluster_deck = cluster_keywords_with_graph(final_deck, n_clusters)
        logger.info(f"Successfully built deck of {len(final_deck)} keywords ...")

        return final_deck, cluster_deck

    def save_results(self, packs: dict):
        """Save the deck to a file."""
        output_dir = "packs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_filename = (
            f"deck_{self.topic}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, "w") as f:
            json.dump(packs, f, indent=4, ensure_ascii=False)
        logger.info(f"Deck saved to {output_path}")

    def run(self):
        """Build a deck of cards from a document."""

        # Generate initial deck
        init_deck = self.batch_gen(self.gen_model)

        # Filter and cluster keywords
        final_deck, cluster_deck = self.filter_and_cluster_keywords(
            init_deck, self.n_clusters
        )

        # Save the packs
        packs = {
            "info": {
                "gen_model": self.gen_model,
                "topic": self.ind_docs_config.get(self.topic, "name"),
                "document_path": self.docs_dir,
                "init_deck_size": len(init_deck),
                "final_deck_size": len(final_deck),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "deck": {
                "init_deck": init_deck,
                "final_deck": final_deck,
                "cluster_deck": cluster_deck,
            },
        }
        self.save_results(packs)

        return packs
