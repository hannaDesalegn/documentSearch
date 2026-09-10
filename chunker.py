def chunk_text(text: str, chunk_size: int = 200) -> list[str]:
    """Split text into a list of chunks of up to chunk_size words each.

    The last chunk may hold fewer than chunk_size words.
    """
    # split() with no argument splits on ANY run of whitespace (spaces, tabs,
    # newlines) and drops empty pieces. That matters for two reasons:
    #   - documents are full of newlines, and we want words, not lines
    #   - "" and "   " both become [], so empty input produces zero chunks
    #     instead of one chunk containing an empty string
    words = text.split()

    chunks = []

    # Walk the word list in steps of chunk_size: 0, 200, 400, ...
    # Each step takes the slice words[i:i + chunk_size]. Python slicing stops
    # at the end of the list, so the final slice is simply whatever is left.
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        # Join the words back into a single string with single spaces.
        chunks.append(" ".join(chunk_words))

    return chunks
