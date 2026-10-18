"""Offline and OpenAI embedding providers for the FAQ vector store."""

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "can", "does", "for",
    "from", "have", "has", "how", "i", "in", "is", "it", "me", "of", "on", "or",
    "the", "this", "to", "what", "when", "where", "which", "who", "with", "would",
    "you", "can", "could", "do", "does", "did", "much", "listed", "product",
    "products", "laptop", "laptops", "computer", "computers",
}
TERM_ALIASES = {
    "cpu": "processor",
    "memory": "ram",
    "life": "battery",
    "lasting": "battery",
    "drive": "storage",
}


def content_terms(text: str) -> set[str]:
    """Return normalized content words for safe, conservative evidence checks."""
    return {
        TERM_ALIASES.get(token, token)
        for token in TOKEN_PATTERN.findall(text.lower())
        if token not in STOP_WORDS
    }


class LocalHashEmbeddings(Embeddings):
    """Small deterministic feature-hash embeddings requiring no model download.

    This local provider is intended for offline demonstrations and tests. It is
    not a substitute for a trained semantic embedding model in production.
    """

    dimensions: int = 512

    def _embed(self, text: str) -> list[float]:
        tokens = sorted(content_terms(text))
        features = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
        vector = [0.0] * self.dimensions
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimensions
            vector[index] += 1.0
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude:
            vector = [value / magnitude for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)