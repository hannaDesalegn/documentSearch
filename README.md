# python-document-search

Small starter project that loads `.txt` files from `documents/`, splits them
into chunks, and ranks those chunks against a question. It can rank by shared
words or by meaning.

## Setup

```bash
pip install -r requirements.txt
```

Semantic search needs an OpenAI API key in the environment:

```bash
export OPENAI_API_KEY=your-key-here
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

## How it works

Each document is split into chunks of up to 200 words.

Keyword search counts how often the question's words appear in each chunk,
ignoring very common words like "the" and "a". It only finds the exact words
you typed, so a question about a car will not match a passage about an
automobile.

Semantic search turns each chunk and the question into a vector with the
`text-embedding-3-small` model, then ranks chunks by cosine similarity. That
finds passages about the same subject even when they share no words.

Chunk vectors are cached in `embeddings_cache.json`, keyed by the chunk text,
so unchanged documents are only ever paid for once. The cache records which
model wrote it and is discarded if the model changes, because vectors from
two different models cannot be compared.

## Tests

```bash
python -m unittest tests.test_main tests.test_chunker tests.test_search tests.test_embeddings
```

The tests never call the OpenAI API.
