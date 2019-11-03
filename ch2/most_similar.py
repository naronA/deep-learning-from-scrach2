"""Most similar."""
import sys

from common.util import preprocess, create_co_matrix

sys.path.append("..")

text = "You say goodbye and I say hello."

corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
