import os
from os import getenv
import aiohttp
from qdrant_client import QdrantClient


os.environ["TOKENIZERS_PARALLELISM"] = "false"


async def encode_query(query_clean: str) -> list[float]:
    embedder_endpoint = f"{getenv('VLLM_EMB_BASE_API')}/embeddings"

    headers = {"Content-Type": "application/json"}

    # Define the data payload
    data = {"model": "elderberry17/USER-bge-m3-x5-sentence", "input": [query_clean]}

    print("starting session...")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            embedder_endpoint, headers=headers, json=data
        ) as response:
            print(response.status)
            raw = await response.json()
            if response.status == 200:
                query_embedding = raw.get("data", [])[0].get("embedding", [])
                return query_embedding
            else:
                print(response.status)
                print(response.content)


def retrieve_points(query_embedding: list[float]):
    qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"))

    search_result = qdrant_client.query_points(
        collection_name=getenv("QDRANT_COLLECTION_NAME"),
        limit=getenv("QDRANT_TOP_N"),
        query=query_embedding,
        with_payload=True,
    ).points

    return search_result


def process_points(points: list[dict]) -> list[tuple[str, str]]:
    qa_tuples = [
        (point.payload["question_clear"], point.payload["content_clear"])
        for point in points
    ]
    return qa_tuples


async def retriever(query_clean: str) -> list[tuple[str, str]]:
    query_embedding = await encode_query(query_clean)
    search_result = retrieve_points(query_embedding)
    search_result_clear = process_points(search_result)

    return search_result_clear
