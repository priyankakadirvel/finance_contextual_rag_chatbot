# preprocessing/chunker.py
import os
from typing import List

def simple_split_text(text: str, max_words: int = 200, overlap: int = 50) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+max_words]
        chunks.append(" ".join(chunk))
        i += max_words - overlap if max_words - overlap > 0 else max_words
    return [c.strip() for c in chunks if c.strip()]
