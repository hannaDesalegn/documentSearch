# python-document-search

Small starter project that loads `.txt` files from `documents/`, splits them
into chunks, and ranks those chunks against a question.

## Run

List every document and its chunk count:

```bash
python main.py
```

Search the documents:

```bash
python main.py how does chunking split a document
```

Each document is split into chunks of up to 200 words. A search counts how
often the question's words appear in each chunk, ignoring very common words
like "the" and "a", and prints the highest scoring chunks with their source
file.

Put `.txt` files in the `documents/` folder to have them loaded.

## Tests

```bash
python -m unittest tests.test_main tests.test_chunker tests.test_search
```
