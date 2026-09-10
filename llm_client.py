import os

from google import genai


# The built client is kept here so every call reuses the same one. That is
# partly to avoid setting up a fresh connection pool on every search, and
# partly because of a bug this project hit for real: writing
# get_client().models.embed_content(...) left nothing holding the client, so
# Python collected it and closed its connection pool while the request was
# still in flight. The request then failed with "client has been closed".
# A module level reference means the client outlives any single call.
_client: genai.Client | None = None


def get_client() -> genai.Client:
    """Build, or reuse, a Gemini client using the GEMINI_API_KEY variable."""
    global _client

    if _client is not None:
        return _client

    # The key is read here rather than when the module is imported. Building
    # the client at import time would make this file impossible to import
    # without a key, which would break the tests and the keyword search too.
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Export the key before running a "
            "semantic search or generating an answer, and never write it "
            "into a source file."
        )

    _client = genai.Client(api_key=api_key)
    return _client
