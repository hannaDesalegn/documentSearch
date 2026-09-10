# python-document-search

Small starter project that loads `.txt` files from `documents/` and prints their names and contents.

## Run

```bash
python main.py
```

Put `.txt` files in the `documents/` folder to see them printed in the terminal.

Each document is also split into chunks of up to 200 words, and the chunk count is printed.

## Tests

```bash
python -m unittest tests.test_main tests.test_chunker
```
