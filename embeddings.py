import json
from pathlib import Path

import numpy as np

# Both this file and answer.py need a client built the same careful way, so
# the setup lives in one place instead of being copied into each of them.
from openai_client import get_client


MODEL = "text-embedding-3-small"
CACHE_PATH = Path(__file__).parent / "embeddings_cache.json"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Turn several pieces of text into vectors with one API call."""
    # The API accepts a list, so one request covers every text. Sending them
    # one at a time would be the same number of tokens but many times the
    # waiting, because each request pays the network round trip again.
    response = get_client().embeddings.create(model=MODEL, input=texts)

    # The response comes back in the same order as the input list.
    return [item.embedding for item in response.data]


def embed_text(text: str) -> list[float]:
    """Turn one piece of text into a vector."""
    return embed_texts([text])[0]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Return how similar two vectors are, from -1.0 to 1.0."""
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)

    # Cosine similarity is the angle between two vectors, not the distance
    # between their tips. Dividing by both lengths cancels magnitude out, so
    # a long passage and a short question about the same subject still score
    # as similar. That is exactly what raw word counting could not do.
    lengths = np.linalg.norm(a) * np.linalg.norm(b)

    # A vector of all zeros has no direction, so there is no angle to
    # measure. Without this guard the division produces nan, and nan quietly
    # loses every comparison during sorting instead of raising an error.
    if lengths == 0:
        return 0.0

    return float(np.dot(a, b) / lengths)


def load_cache() -> dict[str, list[float]]:
    """Read the saved vectors, or start empty if there are none to read."""
    if not CACHE_PATH.exists():
        return {}

    cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    # Vectors from one model mean nothing to another, so a cache written by a
    # different model is thrown away rather than mixed in. Comparing vectors
    # from two models produces numbers that look fine and rank nonsense.
    if cached.get("model") != MODEL:
        return {}

    return cached.get("vectors", {})


def save_cache(vectors: dict[str, list[float]]) -> None:
    """Write the vectors to disk so the next run does not pay for them again."""
    cached = {"model": MODEL, "vectors": vectors}
    CACHE_PATH.write_text(json.dumps(cached), encoding="utf-8")


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Return a vector for each chunk, reusing saved vectors where possible."""
    vectors = load_cache()

    # Collect the chunks that have never been embedded. Identical text is
    # only listed once, because embedding the same string twice would cost
    # twice as much and return the same numbers.
    missing = []
    for chunk in chunks:
        if chunk not in vectors and chunk not in missing:
            missing.append(chunk)

    if missing:
        for chunk, vector in zip(missing, embed_texts(missing)):
            vectors[chunk] = vector
        save_cache(vectors)

    # Look every chunk up by its own text, so the returned list lines up with
    # the chunks that were passed in, cached and freshly embedded alike.
    return [vectors[chunk] for chunk in chunks]


def semantic_search(
    question: str, chunks: list[str], top_k: int = 3
) -> list[tuple[int, int, str]]:
    """Rank chunks by meaning rather than by shared words.

    Returns (score, index, chunk) tuples in the same shape as search() in
    search.py, so the two can be swapped without changing the caller.
    """
    if not chunks:
        return []

    question_vector = embed_text(question)
    chunk_vectors = embed_chunks(chunks)

    results = []
    for index, (chunk, vector) in enumerate(zip(chunks, chunk_vectors)):
        score = cosine_similarity(question_vector, vector)
        results.append((score, index, chunk))

    # Unlike the keyword search, nothing is dropped for scoring zero. Every
    # chunk has some similarity to every question, so the ranking is what
    # carries the meaning here, not the presence of a match.
    results.sort(key=lambda result: (-result[0], result[1]))

    return results[:top_k]
