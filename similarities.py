from sklearn.metrics.pairwise import cosine_similarity
from gensim import matutils as mu

'''
#
# Returns the cosine similarity between vector and feature_vectors.
#
'''
def find_similarities(vector, feature_vectors):
  return cosine_similarity(vector, feature_vectors)


# predict label vector from labeld vectors using cosine similariy
#
# vector: vector to be predicted
# labeld_vectors: is a set of vectors that have been labeled for similarity comparision

def predict(vector, labeled_vectors):

  max = -2
  most_similarity = None

  for v in labeled_vectors:

    distance = mu.cossim(vector, v)
    if distance > max:
      max = distance
      most_similarity = v

  return most_similarity