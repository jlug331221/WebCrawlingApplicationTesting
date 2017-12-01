from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models, similarities
'''
#
# Returns the cosine similarity between vector and feature_vectors.
#
'''
def find_similarities(vector, feature_vectors):
  return cosine_similarity(vector, feature_vectors)

# feature_vector = ['text1', 'texxt2', 'text3']
def predict_LDA(feature_vector):

  base_saved_output = 'training_set_lda_model'

  dictionary = corpora.Dictionary.load_from_text(base_saved_output + '/training_set_dictionary.txt')
  corpus = corpora.MmCorpus(base_saved_output + '/training_corpus.mm')

  model_lda = models.LdaModel.load(base_saved_output + '/LDA.model')

  vec_bow = dictionary.doc2bow(feature_vector)
  vec_lda = model_lda[vec_bow]

  index = similarities.MatrixSimilarity(model_lda[corpus], num_features=46)
  # compare similarity between vec_lda and the rest of the corpus
  sims = index[vec_lda]
  # sort the sims descending order
  sims = sorted(enumerate(sims), key=lambda item: -item[1])

  most_similar = sims[0]

  print(most_similar)

  my_topic = list(dictionary.token2id)[list(dictionary.token2id.values()).index(most_similar[0]+1)]

  print("predicted topic:", my_topic)

  return my_topic

predict_LDA( ['email'])
# predict_LDA( ['email', 'email', 'email', '35'])
# predict_LDA( ['email', 'text', 'email', 'email', '35'])