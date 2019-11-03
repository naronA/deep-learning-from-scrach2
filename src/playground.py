"""playground."""
# flake8: noqa

#%% [markdown]
## Pythonによるコーパスの下準備

#%%[markdown]

### 単語の分散表現

# 世の中には様々な表現方法がある
# - 「コバルトブルー」「シンクレッド」など固有の名前をつける.
# - RGBの3成分がどれだけ存在するかで表現する.

# RGBはベクトル表現
# 深緋 = (201, 23, 30) 赤系の色だということがわかる
# 似た色かどうかもベクトル表現のほうがわかりやすい

# RGBは色の分散表現
# コンパクトで理にかなったベクトル表現
# 単語の意味を的確に捉えたベクトル表現

### 分布仮設
# 自然言語処理の歴史において、単語をベクトルで表す県境は数多く行われてきました。
# 共通のアイディアは「単語の意味は、周囲の単語によって形成される
# これを分布仮設(distributional hypothesis)と呼ばれる
# 単語自体に意味はなく、その単語のコンテキストによって単語の意味が形成される
# - I dring beer. We drink wine. のようにdrinkのちかくには　飲み物が表れやすい
# - I guzzle beer. We guzzle wine.ではguzzleという単語がdrinkと同じような文脈で使われることがわかる
# drinkとguzzleは似ている(guzzleはガブガブ飲む)
# 単語の周囲に存在する単語は関連度が高いので窓で抽出する(windowサイズ2)

#%% [markdown]
### 共起行列
# 分布仮設に基づいて単語をペクトルで表す方法を考える。
# 素直な方法は、周囲の単語を*カウントする*ことです

#%%
import sys

sys.path.append("..")
import numpy as np
from common.util import preprocess

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)
print(id_to_word)

#%% [markdown]

# 語彙数が全部で7個。それぞれの単語について
# コンテキストに含まれる単語の頻度を数えていく
# 全ての単語に対して共起する単語をテーブルにまとめたものを
# 共起行列といいます。

#%%
import numpy as np

C = np.array(
    [
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
    ],
    dtype=np.int32,
)

print(C[0])
print(C[word_to_id["goodbye"]])


#%% [markdown]
# ###ベクトル間の類似度
# 共起行列をもちいてベクトル感の類似度を計測する
# ベクトルの内積・ユークリッド距離などがありますが、単語のベクトル表現の類似度は
# コサイン類似度(cosine similarity)がよく用いられる。
# $$
# similarity(x, y) = \frac{xy}{||x|| ||y||} = \frac{x_1y_1 + ... + x_ny_n}{\sqrt{x_1^2+ .. + x_n^2} \sqrt{y_1^2 + ... + y_n^2}}
# $$


#%%
from common import util

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = util.create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id["you"]]  # youの単語ベクトル
c1 = C[word_to_id["i"]]  # iの単語ベクトル
print(c0, c1)
print(util.cos_similarity(c0, c1))

#%%
import numpy as np

x = np.array([100, -20, 2])
print(x.argsort())
