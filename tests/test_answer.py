from types import SimpleNamespace
import unittest
from unittest.mock import patch

import answer
from answer import NO_RESULTS_ANSWER, SYSTEM_PROMPT, build_prompt, generate_answer


class FakeClient:
    """Stands in for the Gemini client and records what it was asked."""

    def __init__(self, reply: str = "an answer [1]") -> None:
        self.reply = reply
        self.calls: list[dict] = []
        self.models = SimpleNamespace(generate_content=self.generate_content)

    def generate_content(self, model: str, contents: str, config) -> SimpleNamespace:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(text=self.reply)


# Ranked results, deliberately not in document order, with scores that should
# never reach the model.
RESULTS = [
    (4, 2, "Chunking splits a document into smaller pieces."),
    (1, 0, "This is a sample document."),
]


class BuildPromptTests(unittest.TestCase):
    def test_numbers_chunks_starting_at_one(self) -> None:
        prompt = build_prompt("what is chunking", RESULTS)

        self.assertIn("[1] Chunking splits a document into smaller pieces.", prompt)
        self.assertIn("[2] This is a sample document.", prompt)
        self.assertNotIn("[0]", prompt)

    def test_numbers_follow_ranked_order_not_document_order(self) -> None:
        prompt = build_prompt("what is chunking", RESULTS)

        # The top result is chunk index 2, and it must still be numbered [1],
        # because the number is a position in the answer's source list.
        self.assertLess(prompt.index("[1] Chunking"), prompt.index("[2] This is"))

    def test_includes_the_question(self) -> None:
        self.assertIn("Question: what is chunking", build_prompt("what is chunking", RESULTS))

    def test_leaves_the_scores_out(self) -> None:
        # A score is meaningless to the model and would only invite it to
        # trust a chunk more because a number next to it looked large.
        prompt = build_prompt("what is chunking", RESULTS)

        self.assertNotIn("score", prompt.lower())

    def test_single_result_is_numbered_one(self) -> None:
        self.assertIn("[1] only chunk", build_prompt("q", [(1, 0, "only chunk")]))


class SystemPromptTests(unittest.TestCase):
    def test_asks_for_bracket_citations(self) -> None:
        self.assertIn("[1]", SYSTEM_PROMPT)

    def test_allows_refusing_to_answer(self) -> None:
        self.assertIn("cannot answer", SYSTEM_PROMPT)


class GenerateAnswerTests(unittest.TestCase):
    def test_returns_the_models_reply(self) -> None:
        client = FakeClient("Chunking splits documents [1]")

        with patch.object(answer, "get_client", lambda: client):
            reply = generate_answer("what is chunking", RESULTS)

        self.assertEqual("Chunking splits documents [1]", reply)

    def test_sends_the_instructions_and_the_prompt(self) -> None:
        client = FakeClient()

        with patch.object(answer, "get_client", lambda: client):
            generate_answer("what is chunking", RESULTS)

        self.assertEqual(1, len(client.calls))
        call = client.calls[0]
        self.assertEqual("gemini-2.5-flash", call["model"])
        self.assertEqual(build_prompt("what is chunking", RESULTS), call["contents"])

        # The rules travel as a system instruction, kept apart from the
        # chunks rather than pasted above them.
        self.assertEqual(SYSTEM_PROMPT, call["config"].system_instruction)

    def test_no_results_answers_without_calling_the_api(self) -> None:
        client = FakeClient()

        with patch.object(answer, "get_client", lambda: client):
            reply = generate_answer("what is chunking", [])

        self.assertEqual(NO_RESULTS_ANSWER, reply)
        self.assertEqual([], client.calls, "the model was asked with no sources")


if __name__ == "__main__":
    unittest.main()
