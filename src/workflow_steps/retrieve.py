from llama_index.core.workflow import Context
from src.workflow_events import PreprocessEvent, RetrieveEvent
from src.settings import settings

import aiohttp
from qdrant_client import QdrantClient

async def encode_query(query_clean: str) -> list[float]:
    embedder_endpoint = f"{settings.embeddings.BASE_API}/embeddings"
    headers = {"Content-Type": "application/json"}
    data = {"model": settings.embeddings.MODEL, "input": [query_clean]}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            embedder_endpoint, headers=headers, json=data
        ) as response:
            raw = await response.json()
            if response.status == 200:
                return raw.get("data", [])[0].get("embedding", [])
            raise Exception(f"Embeddings API error: {response.status} - {raw}")

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
    
    # Get clear_history from context
    clear_history = await ctx.get("clear_history")
    
    # Get last 2 user messages from clear_history
    last_2_user_messages = [msg for msg in clear_history if msg["role"] == "user"][-2:]
    
    # Concatenate last 2 user messages with current query
    concatenated_query = "\n".join([msg["content"] for msg in last_2_user_messages] + [query_clean])
    print("Query to retrieve:", concatenated_query)
    
    qa = await retriever(concatenated_query)
    return RetrieveEvent(qa=qa)
