from sklearn.metrics.pairwise import cosine_similarity

'''
#
# Returns the cosine similarity between vector and feature_vectors.
#
'''
def find_similarities(vector, feature_vectors):
  return cosine_similarity(vector, feature_vectors)