# python-document-search

Small starter project that loads `.txt` files from `documents/`, splits them
into chunks, ranks those chunks against a question, and can have a language
model write the answer from the chunks it found.

## Setup

```bash
pip install -r requirements.txt
```

Semantic search and generated answers need a Gemini API key in the
environment:

```bash
export GEMINI_API_KEY=your-key-here
```

## Run

List every document and its chunk count:

```bash
python main.py
```

Keyword search, which needs no key and no packages:

```bash
python main.py how does chunking split a document
```

Semantic search, which ranks by meaning:

```bash
python main.py --semantic what is a car
```

A written answer instead of a list of chunks:

```bash
python main.py --answer how does chunking split a document
python main.py --semantic --answer what is a car
```

## How it works

Each document is split into chunks of up to 200 words.

Keyword search counts how often the question's words appear in each chunk,
ignoring very common words like "the" and "a". It only finds the exact words
you typed, so a question about a car will not match a passage about an
automobile.

Semantic search turns each chunk and the question into a vector with the
`gemini-embedding-001` model, then ranks chunks by cosine similarity. That
finds passages about the same subject even when they share no words.

Chunk vectors are cached in `embeddings_cache.json`, keyed by the chunk text,
so unchanged documents are only ever paid for once. The cache records which
model wrote it and is discarded if the model changes, because vectors from
two different models cannot be compared.

With `--answer`, the ranked chunks are numbered and handed to `gemini-2.5-flash`,
which is told to answer using only those chunks, to cite them in brackets
like `[1]`, and to say plainly when the chunks do not contain the answer
rather than guessing. The bracket numbers are positions in the result list,
so the printed `Sources:` section maps each one back to its file.

## Tests

```bash
python -m unittest discover
```

The tests never call the Gemini API.
