import sys
from pathlib import Path

from chunker import chunk_text
from search import search


DOCUMENTS_DIR = Path(__file__).parent / "documents"

# Command line switches, kept in one place so they can be stripped out of the
# question no matter where they were typed.
FLAGS = {"--semantic", "--answer"}


def get_text_files(documents_dir: Path) -> list[Path]:
    return sorted(documents_dir.glob("*.txt"))


def load_chunks(documents_dir: Path) -> list[tuple[str, str]]:
    """Return one (file name, chunk) pair for every chunk of every document."""
    # Search works over one flat list of chunks, not a list of documents, so
    # each chunk has to carry the name of the file it came from. Otherwise a
    # result could be shown but not traced back to its source.
    chunks = []

    for file_path in get_text_files(documents_dir):
        content = file_path.read_text(encoding="utf-8")
        for chunk in chunk_text(content):
            chunks.append((file_path.name, chunk))

    return chunks


def print_documents(documents_dir: Path) -> None:
    text_files = get_text_files(documents_dir)

    if not text_files:
        print("No .txt files found in documents/")
        return

    for file_path in text_files:
        content = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(content)
        print(f"{file_path.name}")
        print(content)
        print(f"chunks: {len(chunks)}")
        print()


def run_search(
    question: str, chunk_texts: list[str], use_semantic: bool
) -> list[tuple[float, int, str]]:
    """Rank the chunks with whichever search the flags asked for."""
    if use_semantic:
        # Imported here, not at the top of the file, so that the keyword
        # search still runs on a machine with no numpy, no google-genai package
        # and no API key. Only the semantic path needs any of those.
        from embeddings import semantic_search

        return semantic_search(question, chunk_texts)

    return search(question, chunk_texts)


def format_score(score: float | int) -> str:
    """Keyword scores are whole counts, similarity scores are fractions."""
    return f"{score:.3f}" if isinstance(score, float) else str(score)


def print_search_results(
    documents_dir: Path, question: str, use_semantic: bool = False
) -> None:
    chunks = load_chunks(documents_dir)

    if not chunks:
        print("No .txt files found in documents/")
        return

    # The search only needs the text, so hand it the chunk text alone and keep
    # the index it returns to look the file name back up afterwards.
    chunk_texts = [chunk for _, chunk in chunks]
    results = run_search(question, chunk_texts, use_semantic)

    if not results:
        print(f"No chunks matched: {question}")
        return

    print(f"Results for: {question}")
    print()

    for score, index, chunk in results:
        source_name = chunks[index][0]
        print(f"score {format_score(score)} - {source_name}")
        print(chunk)
        print()


def print_answer(
    documents_dir: Path, question: str, use_semantic: bool = False
) -> None:
    from answer import generate_answer

    chunks = load_chunks(documents_dir)

    if not chunks:
        print("No .txt files found in documents/")
        return

    chunk_texts = [chunk for _, chunk in chunks]
    results = run_search(question, chunk_texts, use_semantic)

    print(generate_answer(question, results))

    if not results:
        return

    # The model cites chunks by the number they were given in the prompt, and
    # that number is just the position in the results list. Numbering the same
    # way here is what turns a bracket in the answer back into a file name the
    # reader can go and check.
    print()
    print("Sources:")
    for number, (_, index, _) in enumerate(results, start=1):
        source_name = chunks[index][0]
        print(f"[{number}] {source_name}")


if __name__ == "__main__":
    arguments = sys.argv[1:]

    # Pull the flags out wherever they appear, so they can be written before
    # or after the question. Everything left over is the question itself,
    # which means a multi-word question works without quotes.
    use_semantic = "--semantic" in arguments
    use_answer = "--answer" in arguments
    question = " ".join(word for word in arguments if word not in FLAGS)

    if not question:
        print_documents(DOCUMENTS_DIR)
    elif use_answer:
        print_answer(DOCUMENTS_DIR, question, use_semantic)
    else:
        print_search_results(DOCUMENTS_DIR, question, use_semantic)
