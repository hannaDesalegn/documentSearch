from pathlib import Path

from chunker import chunk_text


DOCUMENTS_DIR = Path(__file__).parent / "documents"


def get_text_files(documents_dir: Path) -> list[Path]:
    return sorted(documents_dir.glob("*.txt"))


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


if __name__ == "__main__":
    print_documents(DOCUMENTS_DIR)
