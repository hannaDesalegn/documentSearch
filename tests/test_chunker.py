import unittest

from chunker import chunk_text


class ChunkTextTests(unittest.TestCase):
    def test_splits_long_text_into_multiple_chunks(self) -> None:
        text = " ".join(f"word{number}" for number in range(450))

        chunks = chunk_text(text, chunk_size=200)

        self.assertEqual(3, len(chunks))
        self.assertEqual(200, len(chunks[0].split()))
        self.assertEqual(200, len(chunks[1].split()))
        self.assertEqual(50, len(chunks[2].split()))

    def test_empty_text_produces_no_chunks(self) -> None:
        self.assertEqual([], chunk_text(""))

    def test_text_shorter_than_chunk_size_produces_one_chunk(self) -> None:
        chunks = chunk_text("a short handful of words", chunk_size=200)

        self.assertEqual(["a short handful of words"], chunks)


if __name__ == "__main__":
    unittest.main()
