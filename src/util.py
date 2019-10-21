"""util."""
import numpy as np
from numpy import ndarray
from typing import List, Dict, Tuple


def preprocess(text: str) -> Tuple[ndarray, Dict[str, int], Dict[int, str]]:
    """Preprocess test."""
    text = text.lower()
    text = text.replace('.', ' .')
    words: List[str] = text.split()
    word_to_id: Dict[str, int] = {}
    id_to_word: Dict[int, str] = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return (corpus, word_to_id, id_to_word)
