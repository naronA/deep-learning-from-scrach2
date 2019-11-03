"""util."""
from typing import Dict, List, Tuple

import numpy as np
from numpy import ndarray


def preprocess(text: str) -> Tuple[ndarray, Dict[str, int], Dict[int, str]]:
    """Preprocess test."""
    text = text.lower()
    text = text.replace(".", " .")
    words: List[str] = text.split()
    word_to_id: Dict[str, int] = {}
    id_to_word: Dict[int, str] = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus: np.ndarray = np.array([word_to_id[w] for w in words])

    return (corpus, word_to_id, id_to_word)


def create_co_matrix(corpus, vocab_size, window_size=1) -> np.ndarray:
    """Create co matrix."""
    corpus_size: int = len(corpus)
    co_matrix: np.ndarray = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def cos_similarity(x, y, eps=1e-8):
    """Cosine similarity."""
    nx = x / np.sqrt(np.sum(x ** 2) + eps)  # xの正規化
    ny = y / np.sqrt(np.sum(y ** 2) + eps)  # yの正規化
    return np.dot(nx, ny)


def most_similar(
    query: str, word_to_id: Dict[str, int], id_to_word: Dict[int, str], word_matrix: np.ndarray, top: int = 5
):
    """最も似た単語を返す."""
    # 1. クエリを取り出す (クエリ=単語)
    if query not in word_to_id:
        print("%s is not found" % query)
        return
    print("\n[query] " + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # コサイン類似度の算出
    # 単語の数だけ
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # コサイン類似度の結果から、その値を高い順に出力
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(" %s: %s" % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return
