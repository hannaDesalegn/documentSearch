import unittest

from search import score_chunk, search, tokenize


class TokenizeTests(unittest.TestCase):
    def test_lowercases_and_drops_punctuation(self) -> None:
        self.assertEqual(["a", "document"], tokenize("A Document."))

    def test_empty_text_produces_no_words(self) -> None:
        self.assertEqual([], tokenize(""))


class ScoreChunkTests(unittest.TestCase):
    def test_counts_every_occurrence_of_a_question_word(self) -> None:
        self.assertEqual(3, score_chunk("chunk", "chunk chunk chunk"))

    def test_ignores_words_that_only_contain_the_query_word(self) -> None:
        # "stores" and "restore" contain "store" but are different words.
        self.assertEqual(0, score_chunk("store", "stores and restore"))

    def test_ignores_common_stop_words(self) -> None:
        self.assertEqual(0, score_chunk("the and a", "the cat and a hat"))

    def test_unmatched_question_scores_zero(self) -> None:
        self.assertEqual(0, score_chunk("embeddings", "keyword search only"))


class SearchTests(unittest.TestCase):
    def test_ranks_chunks_from_most_to_least_relevant(self) -> None:
        chunks = ["one search", "search search search", "search search"]

        results = search("search", chunks)

        self.assertEqual([3, 2, 1], [score for score, _, _ in results])
        self.assertEqual([1, 2, 0], [index for _, index, _ in results])

    def test_leaves_out_chunks_that_share_no_words(self) -> None:
        chunks = ["chunking splits documents", "nothing relevant here"]

        results = search("chunking", chunks)

        self.assertEqual(1, len(results))
        self.assertEqual((1, 0, "chunking splits documents"), results[0])

    def test_returns_at_most_top_k_results(self) -> None:
        chunks = ["search", "search", "search", "search"]

        results = search("search", chunks, top_k=2)

        self.assertEqual(2, len(results))

    def test_breaks_ties_in_document_order(self) -> None:
        chunks = ["search", "search", "search"]

        results = search("search", chunks)

        self.assertEqual([0, 1, 2], [index for _, index, _ in results])

    def test_question_with_no_meaningful_words_matches_nothing(self) -> None:
        self.assertEqual([], search("the and of", ["the cat and the hat"]))

    def test_no_chunks_returns_no_results(self) -> None:
        self.assertEqual([], search("anything", []))


if __name__ == "__main__":
    unittest.main()
