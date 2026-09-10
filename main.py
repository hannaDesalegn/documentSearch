import sys
from pathlib import Path

from chunker import chunk_text
from search import search


DOCUMENTS_DIR = Path(__file__).parent / "documents"


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


def print_search_results(documents_dir: Path, question: str) -> None:
    chunks = load_chunks(documents_dir)

    if not chunks:
        print("No .txt files found in documents/")
        return

    # search() only needs the text, so hand it the chunk text alone and keep
    # the index it returns to look the file name back up afterwards.
    chunk_texts = [chunk for _, chunk in chunks]
    results = search(question, chunk_texts)

    if not results:
        print(f"No chunks matched: {question}")
        return

    print(f"Results for: {question}")
    print()

    for score, index, chunk in results:
        source_name = chunks[index][0]
        print(f"score {score} - {source_name}")
        print(chunk)
        print()


if __name__ == "__main__":
    # Everything after the script name is the question, so a multi-word
    # question works without quotes as well as with them.
    question = " ".join(sys.argv[1:])

    if question:
        print_search_results(DOCUMENTS_DIR, question)
    else:
        print_documents(DOCUMENTS_DIR)
