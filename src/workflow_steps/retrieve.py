from llama_index.core.workflow import Context
from src.workflow_events import PreprocessEvent, RetrieveEvent
from src.settings import settings

import aiohttp
from qdrant_client import QdrantClient

async def encode_query(query_clean: str) -> list[float]:
    embedder_endpoint = f"{settings.embeddings.BASE_API}/embeddings"

    headers = {"Content-Type": "application/json"}

    # Define the data payload
    data = {"model": settings.embeddings.MODEL, "input": [query_clean]}

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
    qdrant_client = QdrantClient(url=settings.qdrant.URL)

    search_result = qdrant_client.query_points(
        collection_name=settings.qdrant.COLLECTION_NAME,
        limit=settings.qdrant.TOP_N,
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

async def retrieve_step(ev: PreprocessEvent, ctx: Context) -> RetrieveEvent:
    query_clean = ev.query_clean
    await ctx.set("query_clean", query_clean)  # Saving clean query for use later

    qa = await retriever(query_clean)
    return RetrieveEvent(qa=qa)
