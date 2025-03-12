import json
import requests


def retriever(query_clean: str) -> list[tuple[str, str]]:
    '''
    We need to specify endpoints for:
    - embedder model
    - qdrant database
    '''
    
    # TODO retrieve
    # mock values
    VLLM_HOST = "http://localhost:8000"
    embedder_endpoint = f"{VLLM_HOST}/v1/embeddings"

    headers = {
        "Content-Type": "application/json"
    }

    # Define the data payload
    data = {
        "model": "elderberry17/USER-bge-m3-x5",
        "input": [query_clean]
    }

    response = requests.post(embedder_endpoint, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        query_embedding = response.json().get('data', [])[0].get('embedding', [])
        print(f"embedding len: {len(query_embedding)}")
    else:
        print(response.status_code)

    # TODO search similar queries from qdrant