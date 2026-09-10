from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import embeddings
from embeddings import cosine_similarity, embed_chunks, semantic_search


class CosineSimilarityTests(unittest.TestCase):
    def test_identical_vectors_score_one(self) -> None:
        self.assertAlmostEqual(1.0, cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]))

    def test_opposite_vectors_score_minus_one(self) -> None:
        self.assertAlmostEqual(-1.0, cosine_similarity([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]))

    def test_unrelated_vectors_score_zero(self) -> None:
        self.assertAlmostEqual(0.0, cosine_similarity([1.0, 0.0], [0.0, 1.0]))

    def test_length_does_not_change_the_score(self) -> None:
        # Same direction, very different magnitude. Cosine measures the angle
        # only, which is why a long chunk is not favoured over a short one.
        self.assertAlmostEqual(1.0, cosine_similarity([1.0, 1.0], [50.0, 50.0]))

    def test_zero_vector_scores_zero_instead_of_nan(self) -> None:
        self.assertEqual(0.0, cosine_similarity([0.0, 0.0], [1.0, 2.0]))


class EmbedChunksCacheTests(unittest.TestCase):
    """Cache behaviour, checked with a stand-in for the API call."""

    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        cache_path = Path(self.temp_dir.name) / "embeddings_cache.json"
        patcher = patch.object(embeddings, "CACHE_PATH", cache_path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def fake_embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return a predictable vector per text, and record what was asked for."""
        self.embedded.extend(texts)
        return [[float(len(text)), 1.0] for text in texts]

    def test_second_call_reuses_the_saved_vectors(self) -> None:
        chunks = ["first chunk", "second chunk"]

        with patch.object(embeddings, "embed_texts", self.fake_embed_texts):
            self.embedded = []
            first_run = embed_chunks(chunks)

            self.embedded = []
            second_run = embed_chunks(chunks)

        self.assertEqual(first_run, second_run)
        self.assertEqual([], self.embedded, "cached chunks were embedded again")

    def test_only_the_new_chunk_is_embedded(self) -> None:
        with patch.object(embeddings, "embed_texts", self.fake_embed_texts):
            self.embedded = []
            embed_chunks(["first chunk"])

            self.embedded = []
            embed_chunks(["first chunk", "brand new chunk"])

        self.assertEqual(["brand new chunk"], self.embedded)

    def test_repeated_chunk_is_embedded_once(self) -> None:
        with patch.object(embeddings, "embed_texts", self.fake_embed_texts):
            self.embedded = []
            vectors = embed_chunks(["same text", "same text"])

        self.assertEqual(["same text"], self.embedded)
        self.assertEqual(vectors[0], vectors[1])

    def test_cache_from_another_model_is_ignored(self) -> None:
        embeddings.CACHE_PATH.write_text(
            '{"model": "some-older-model", "vectors": {"first chunk": [9.0, 9.0]}}',
            encoding="utf-8",
        )

        with patch.object(embeddings, "embed_texts", self.fake_embed_texts):
            self.embedded = []
            vectors = embed_chunks(["first chunk"])

        self.assertEqual(["first chunk"], self.embedded)
        self.assertNotEqual([9.0, 9.0], vectors[0])


class SemanticSearchTests(unittest.TestCase):
    """Ranking behaviour, checked with plain vectors and no API call."""

    def test_ranks_chunks_by_similarity_to_the_question(self) -> None:
        chunks = ["far away", "exact match", "halfway"]
        vectors = [[-1.0, 0.0], [1.0, 0.0], [1.0, 1.0]]

        with patch.object(embeddings, "embed_text", lambda text: [1.0, 0.0]), \
             patch.object(embeddings, "embed_chunks", lambda texts: vectors):
            results = semantic_search("anything", chunks)

        self.assertEqual([1, 2, 0], [index for _, index, _ in results])
        self.assertAlmostEqual(1.0, results[0][0])
        self.assertAlmostEqual(-1.0, results[2][0])

    def test_top_k_limits_the_results(self) -> None:
        chunks = ["one", "two", "three"]
        vectors = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]

        with patch.object(embeddings, "embed_text", lambda text: [1.0, 0.0]), \
             patch.object(embeddings, "embed_chunks", lambda texts: vectors):
            results = semantic_search("anything", chunks, top_k=2)

        self.assertEqual(2, len(results))

    def test_no_chunks_returns_no_results(self) -> None:
        self.assertEqual([], semantic_search("anything", []))


if __name__ == "__main__":
    unittest.main()
