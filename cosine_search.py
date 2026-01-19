"""Custom cosine similarity search implementation."""

import numpy as np
from numpy.typing import NDArray
from typing import Any


def cosine_similarity(vec1: NDArray[np.float32], vec2: NDArray[np.float32]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity score between -1 and 1.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def cosine_search(
    client: Any,
    query: str,
    user_id: str,
    limit: int = 10,
    threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Custom cosine similarity search using mem0 components.

    Args:
        client: mem0 Memory client instance.
        query: Search query string.
        user_id: User ID to filter results.
        limit: Maximum number of results to return.
        threshold: Minimum similarity score threshold.

    Returns:
        List of results sorted by cosine similarity score.
    """
    # Generate query embedding
    query_embedding = client.embedding_model.embed(query, "search")
    query_vector = np.array(query_embedding, dtype=np.float32)

    # Get all points from vector store for this user
    qdrant_client = client.vector_store.client
    collection_name = client.vector_store.collection_name

    # Scroll through all points with user_id filter
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    points, _ = qdrant_client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        ),
        with_vectors=True,
        with_payload=True,
        limit=1000,
    )

    # Calculate cosine similarity for each point
    results = []
    for point in points:
        if point.vector is None:
            continue

        point_vector = np.array(point.vector, dtype=np.float32)
        score = cosine_similarity(query_vector, point_vector)

        if score >= threshold:
            results.append({
                "id": point.id,
                "memory": point.payload.get("data", ""),
                "score": score,
                "user_id": point.payload.get("user_id"),
                "created_at": point.payload.get("created_at"),
            })

    # Sort by score descending and limit results
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]
