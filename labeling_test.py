from gensim import corpora, models, similarities, matutils
import numpy.random as np_rand
import random as rand
FIXED_SEED = 3

t = [
  ['last', 'name', 'text', 'last', 'name', 'last', 'name', '35'],
  ['email', 'text', 'email', 'email', '35'],
  ['password', 'password', 'password', 'password', '25'],
  ['verify', 'password', 'password', 'check', 'password', 'check', 'password', '25']
]

for i in range(len(t)):
  print(t[i])
print()

dictionary = corpora.Dictionary(t)

print(dictionary.token2id)
print()

b_o_w = [dictionary.doc2bow(feature_vector)
                  for feature_vector in t]

print('BoW counts:')
for i in range(len(b_o_w)):
  print(b_o_w[i])
print()

tfidf_model = models.TfidfModel(b_o_w, id2word=dictionary)

tfidf = tfidf_model[b_o_w]

# print('tfidf model:')
# for tfidf_vect in tfidf:
#   print(str(tfidf_vect))
# print()

np_rand.seed(FIXED_SEED)
rand.seed(FIXED_SEED)
lda_model = models.LdaModel(tfidf.corpus, num_topics=len(tfidf.corpus), id2word=dictionary,
                      passes=10, minimum_probability=0, alpha='auto')

print('Topic BoW models:')
for i in range(len(b_o_w)):
  e = b_o_w[i]
  print(lda_model[e])
print()

tops = set(lda_model.show_topics())

top_clusters = []
for l in tops:
  top = []
  for t in l[1].split(" + "):
    top.append((t.split('*')[0], t.split('*')[1]))
  top_clusters.append(top)

print('Top clusters:')
print(top_clusters)
print()

top_wordonly = []
for i in top_clusters:
  top_wordonly.append(':'.join([j[1] for j in i]))

for i in range(len(top_wordonly)):
  print(top_wordonly[i])
print()

for i in range(len(tfidf.corpus)):
  print(lda_model.print_topic(i, topn=2))
print()

tokens2id = dictionary.token2id

for i in range(len(tfidf.corpus)):
  t = lda_model.get_topic_terms(i, topn=1)

  for x in t:
    t = t, list(tokens2id.keys())[list(tokens2id.values()).index(x[0])]

  print(t)



