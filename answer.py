from google.genai import types

from llm_client import get_client


MODEL = "gemini-2.5-flash"

# The model is told three things, and the third one is the one that matters.
# A language model asked a question it cannot answer will usually invent a
# fluent answer rather than admit the gap, and an invented answer is worse
# than no answer here, because it arrives wearing the same confident tone as
# a real one. Saying so plainly has to be an allowed outcome.
SYSTEM_PROMPT = (
    "You answer questions using only the numbered document chunks the user "
    "provides.\n"
    "\n"
    "Rules:\n"
    "1. Use only information found in the numbered chunks. Do not add facts "
    "from your own knowledge, even if you are confident they are correct.\n"
    "2. After each part of your answer, mark the chunk it came from in "
    "brackets, like [1], or [2][3] when a statement draws on more than one "
    "chunk. Put the brackets directly after the relevant sentence.\n"
    "3. If the chunks do not contain what is needed to answer, say plainly "
    "that you cannot answer the question from the given documents. Do not "
    "guess and do not fill the gap from memory."
)

# Returned without asking the model anything when the search found nothing.
NO_RESULTS_ANSWER = (
    "I cannot answer that question from the given documents, because the "
    "search returned no matching chunks."
)


def build_prompt(question: str, results: list[tuple[float, int, str]]) -> str:
    """Lay the retrieved chunks out as a numbered list followed by the question."""
    # The numbering starts at 1 and follows the order of results, which is
    # ranked order. These numbers are what the model cites in brackets, and
    # they are also the position in this list, so a citation can be mapped
    # back to the chunk and its source file afterwards.
    #
    # The chunk's own score and its index in the document set are deliberately
    # left out. Neither means anything to the model, and showing a score would
    # invite it to treat a high number as a reason to trust a chunk more.
    numbered_chunks = []
    for number, (_, _, chunk) in enumerate(results, start=1):
        numbered_chunks.append(f"[{number}] {chunk}")

    chunk_text = "\n\n".join(numbered_chunks)

    # The question goes last. A model reads the whole prompt, but putting the
    # question after the chunks keeps it from being buried when the chunks are
    # long, and it reads in the order the work happens: here is the evidence,
    # now here is what to do with it.
    return f"Document chunks:\n\n{chunk_text}\n\nQuestion: {question}"


def generate_answer(question: str, results: list[tuple[float, int, str]]) -> str:
    """Ask the model to answer the question from the retrieved chunks."""
    # With no chunks there is no evidence, and a model given an empty list of
    # sources and told to cite them has nothing to do but invent. Answering
    # directly costs nothing and removes the temptation entirely.
    if not results:
        return NO_RESULTS_ANSWER

    # The rules go in as a system instruction rather than at the top of the
    # prompt, so they stay separate from the chunks. A chunk that happens to
    # contain instruction-like wording is then plainly just document text,
    # not something competing with the rules for the model's attention.
    client = get_client()
    response = client.models.generate_content(
        model=MODEL,
        contents=build_prompt(question, results),
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )

    return response.text
