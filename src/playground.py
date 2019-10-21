"""playground."""
# flake8: noqa

#%%
import numpy as np

#%%

x = np.array([1, 2, 3])
x.__class__

#%%
print(x.shape)
print(x.ndim)

#%%
W = np.array([[1, 2, 3], [4, 5, 6]])
print(W.shape)
print(W.ndim)

#%%
import cupy as cp

x = cp.arange(6).reshape(2, 3).astype('f')
print(x)

#%% [markdown]
## Pythonによるコーパスの下準備

#%%
from src import util
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = util.preprocess(text)
print(corpus)
print(word_to_id)
print(id_to_word)


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
# 共通のアイディアは「単語の意味は、周囲の単語によって形成される」
# これを分布仮設(distributional hypothesis)と呼ばれる
# 単語自体に意味はなく、その単語のコンテキストによって単語の意味が形成される
# - I dring beer. We drink wine. のようにdrinkのちかくには　飲み物が表れやすい
# - I guzzle beer. We guzzle wine.ではguzzleという単語がdrinkと同じような文脈で使われることがわかる
# drinkとguzzleは似ている(guzzleはガブガブ飲む)
# 単語の周囲に存在する単語(windowサイズ2)