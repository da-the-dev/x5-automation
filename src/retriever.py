from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient


def encode_query(query_clean: str) -> list[float]:
    # TODO set variables to config and init embedder_model one time
    HF_MODEL_NAME = "elderberry17/USER-bge-m3-x5"
    embedder_model = HuggingFaceEmbedding(model_name=HF_MODEL_NAME)
    query_embedding = embedder_model.get_text_embedding(query_clean)
    return query_embedding


def retrieve_points(query_embedding: list[float]):
    # TODO set variables to config and init client one time
    QDRANT_QUERY_URL = "http://localhost:6333"
    TOPN = 10
    COLLECTION_NAME = "X5_database"

    qdrant_client = QdrantClient(url=QDRANT_QUERY_URL)

    search_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        with_payload=True,
        limit=TOPN
    ).points

    return search_result


def process_points(points: list[dict]) -> list[tuple[str, str]]:
    qa_tuples = [(point['question_clear'], point['content_clear']) for point in points]
    return qa_tuples


def retriever(query_clean: str) -> list[tuple[str, str]]:
    query_embedding = encode_query(query_clean)
    search_result = retrieve_points(query_embedding)
    search_result_clear = process_points(search_result)

    return search_result_clear
    