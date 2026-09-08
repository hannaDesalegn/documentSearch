from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from main import get_text_files, print_documents


class DocumentLoaderTests(unittest.TestCase):
    def test_get_text_files_finds_only_txt_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            documents_dir = Path(temp_dir)
            (documents_dir / "a.txt").write_text("A", encoding="utf-8")
            (documents_dir / "b.md").write_text("B", encoding="utf-8")

            text_files = get_text_files(documents_dir)

            self.assertEqual([documents_dir / "a.txt"], text_files)

    def test_print_documents_handles_empty_directory(self) -> None:
        with TemporaryDirectory() as temp_dir:
            documents_dir = Path(temp_dir)
            output = StringIO()

            with redirect_stdout(output):
                print_documents(documents_dir)

            self.assertEqual("No .txt files found in documents/\n", output.getvalue())


if __name__ == "__main__":
    unittest.main()
