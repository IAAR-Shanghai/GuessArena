import networkx as nx
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger
from sklearn.cluster import SpectralClustering
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def filter_by_sim(topic, keywords, upper_thresh=0.9, lower_thresh=0.1):
    """Filter keywords by cosine similarity to the topic."""
    # Embed the topic
    topic_embedding = EMBEDDING_MODEL.embed_query(topic)
    # Embed the keywords
    keywords_embeddings = [EMBEDDING_MODEL.embed_query(kw) for kw in keywords]
    # Compute cosine similarity
    filtered_keywords = []
    filtered_keywords_embeddings = []
    for kw, kw_emb in zip(keywords, keywords_embeddings):
        sim = cosine_similarity([topic_embedding], [kw_emb])[0][0]
        if sim < upper_thresh and sim > lower_thresh:
            filtered_keywords.append(kw)
            filtered_keywords_embeddings.append(kw_emb)
    logger.info("Successfully filtered keywords by similarity ...")
    return filtered_keywords, filtered_keywords_embeddings


def build_similarity_graph(keywords):
    """Build a similarity graph of keywords."""
    # Embed the keywords
    keywords_embeddings = [EMBEDDING_MODEL.embed_query(kw) for kw in keywords]
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(keywords_embeddings)
    # Build a graph
    G = nx.Graph()
    for idx, keyword in enumerate(keywords):
        G.add_node(keyword, embedding=keywords_embeddings[idx])
    # Add edges
    threshold = 0.6
    for i in range(len(keywords)):
        for j in range(i + 1, len(keywords)):
            if similarity_matrix[i][j] > threshold:
                G.add_edge(keywords[i], keywords[j], weight=similarity_matrix[i][j])

    return G, similarity_matrix


def cluster_keywords_with_graph(keywords, n_clusters=5):
    """Cluster keywords using a graph-based approach."""
    # Build a similarity graph
    G, similarity_matrix = build_similarity_graph(keywords)
    # Cluster keywords
    spectral_clustering = SpectralClustering(
        n_clusters=n_clusters, affinity="precomputed", random_state=0
    )
    labels = spectral_clustering.fit_predict(similarity_matrix)
    # Get cluster keywords
    cluster_keywords = {i: [] for i in range(n_clusters)}
    for i, label in enumerate(labels):
        cluster_keywords[label].append(keywords[i])
    # Return all keywords grouped by clusters
    selected_keywords = {}
    for cluster_idx, cluster_kw in cluster_keywords.items():
        selected_keywords[f"Cluster_{cluster_idx}"] = cluster_kw

    logger.info("Successfully clustered keywords using graph-based approach ...")
    return selected_keywords


def recall_chain_keywords(starting_kw, keywords, max_depth=2):
    """Recall chain keywords based on the starting keyword."""
    # Embed the starting keyword and keywords
    keywords_embeddings = [EMBEDDING_MODEL.embed_query(kw) for kw in keywords]
    # Initialize the recall chain
    recall_chain_keywords = [starting_kw]
    curr_depth = 0
    # Iterate until the max depth is reached
    while curr_depth < max_depth:
        curr_depth += 1
        last_kw = recall_chain_keywords[-1]
        # Compute cosine similarity
        last_kw_embedding = EMBEDDING_MODEL.embed_query(last_kw)
        similarities = {
            kw: cosine_similarity([last_kw_embedding], [kw_emb])[0][0]
            for kw, kw_emb in zip(keywords, keywords_embeddings)
        }
        sorted_similarities = sorted(
            similarities.items(), key=lambda x: x[1], reverse=True
        )
        simlarity_threshold = 0.9
        for kw, sim in sorted_similarities:
            if kw not in recall_chain_keywords and sim < simlarity_threshold:
                recall_chain_keywords.append(kw)
                break

    return recall_chain_keywords


def find_best_k(keywords, max_clusters=10):
    """Find the best number of clusters based on Davies-Bouldin index."""
    # Build a similarity graph
    G, similarity_matrix = build_similarity_graph(keywords)

    # Compute the Davies-Bouldin scores
    db_scores = []
    for n_clusters in range(2, max_clusters + 1):
        spectral_clustering = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed", random_state=0
        )
        labels = spectral_clustering.fit_predict(similarity_matrix)

        # Davies-Bouldin score is calculated based on the data and the predicted labels
        db_score = davies_bouldin_score(similarity_matrix, labels)
        db_scores.append(db_score)

    # Find the best number of clusters (the one with the smallest Davies-Bouldin score)
    best_k = np.argmin(db_scores) + 2  # +2 because the range starts from 2
    logger.info(f"Best number of clusters based on Davies-Bouldin index: {best_k}")
    return best_k
