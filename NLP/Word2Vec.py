import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import gensim.downloader as api
wv= api.load('word2vec-google-news-300')
vec_king = wv['king']
# print(vec_king)

# print(wv.most_similar('cricket'))

# print(wv.similarity('cricket', 'badminton'))

vec = wv['king']-wv['man'] + wv['woman']
print(wv.most_similar([vec]))
