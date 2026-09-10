import re


# Words so common that they say nothing about what a chunk is about. Counting
# them lets a chunk win on the word "a" alone, which is not relevance, it is
# just length. Real search engines work this out from the collection itself
# rather than from a hardcoded list, but the goal is the same: a rare word
# carries more meaning than a common one.
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do",
    "does", "for", "from", "has", "have", "how", "i", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "then", "there", "these", "this", "to",
    "was", "what", "when", "where", "which", "why", "will", "with", "you",
}


def tokenize(text: str) -> list[str]:
    """Break text into lowercase words, dropping punctuation."""
    # findall keeps only runs of letters and digits, so "Document." becomes
    # "document" and the trailing period disappears. Lowercasing here means a
    # question asking "Document" still matches a chunk saying "document".
    #
    # Both the question and the chunks go through this same function. That is
    # the important part: if the two sides are tokenized differently, words
    # that look identical on screen will silently fail to match.
    return re.findall(r"[a-z0-9]+", text.lower())


def remove_stop_words(words: list[str]) -> list[str]:
    """Drop the very common words that carry no meaning of their own."""
    return [word for word in words if word not in STOP_WORDS]


def score_chunk(question: str, chunk: str) -> int:
    """Count how often the question's meaningful words appear in the chunk."""
    # Only the question is filtered. The chunk can keep every word, because
    # the score only ever counts words the question actually asked about.
    question_words = remove_stop_words(tokenize(question))
    chunk_words = tokenize(chunk)

    # Add up one count per question word. A chunk that uses a question word
    # three times scores 3 for that word, so repetition raises the score.
    #
    # Counting whole words matters. Searching the raw string for "the" would
    # also match the "the" hiding inside "there" and "other", which quietly
    # inflates the score of chunks that never discuss the topic at all.
    score = 0
    for word in question_words:
        score += chunk_words.count(word)

    return score


def search(question: str, chunks: list[str], top_k: int = 3) -> list[tuple[int, int, str]]:
    """Rank chunks by relevance to the question.

    Returns a list of (score, index, chunk) sorted from most to least
    relevant. Chunks sharing no words with the question are left out, and at
    most top_k results come back.
    """
    results = []

    for index, chunk in enumerate(chunks):
        score = score_chunk(question, chunk)
        # A score of 0 means the chunk has nothing in common with the
        # question. That is not a weak answer, it is not an answer.
        if score > 0:
            results.append((score, index, chunk))

    # Sort by score, highest first. The minus sign flips the order, because
    # sort() goes smallest to largest by default. Sorting on index second
    # breaks ties in the order the chunks were loaded, so equal scores come
    # back in a stable, predictable order instead of an arbitrary one.
    results.sort(key=lambda result: (-result[0], result[1]))

    return results[:top_k]
