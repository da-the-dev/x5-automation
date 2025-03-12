import torch
import pandas as pd
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct


if __name__ == "__main__":
    
    COLLECTION_NAME = "X5_database"

    vectors = torch.load("embeddings.pt",  weights_only=True)
    df_payload = pd.read_csv("qa_df_pairs_db.csv")
    payloads = df_payload.to_dict(orient="records")

    points = [
        PointStruct(id=i, vector=vectors[i], payload=payloads[i])
        for i in range(len(vectors))
    ]

    client = QdrantClient(url="http://localhost:6333")

    # client.delete_collection(collection_name=COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.DOT),
    )

    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name="X5_database", wait=True, points=batch)
    print("Данные загружены")

    points = client.scroll(collection_name="X5_database", limit=1)
    if points:
        print("Первый элемент:")
        print(points[0][0])
    else:
        print("Список записей пуст")