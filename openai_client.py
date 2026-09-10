import os

from openai import OpenAI


def get_client() -> OpenAI:
    """Build an OpenAI client from the OPENAI_API_KEY environment variable."""
    # The key is read here rather than when the module is imported. Building
    # the client at import time would make this file impossible to import
    # without a key, which would break the tests and the keyword search too.
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export the key before running a "
            "semantic search or generating an answer, and never write it "
            "into a source file."
        )

    return OpenAI(api_key=api_key)
