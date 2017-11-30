from sklearn.metrics.pairwise import cosine_similarity
from gensim import matutils as mu

'''
#
# Returns the cosine similarity between vector and feature_vectors.
#
'''
def find_similarities(vector, feature_vectors):
  return cosine_similarity(vector, feature_vectors)



def get_labeled_vectors():
  lines = open("labeled-vectors.txt")

  labeled_vectors = []
  for line in lines:
    items = line.split(",")
    for item in items:
      item = item.strip()
      item = item.strip('(')
      item = item.strip('[')
      item = item.strip('(')
      item = item.strip(')')
      item = item.strip(']')
      item = item.strip()
      print(item)


def predict_cossim(vector):
  lines = open("labeled-vectors.txt")

  for line in lines:
    print(line)



get_labeled_vectors()


# predict label vector from labeld vectors using cosine similariy
#
# vector: vector to be predicted
# labeled_vectors: is a set of vectors that have been labeled for similarity comparision
#
# return: most similarity vector. From this most similarity we will figure out the class label.

def predict(vector, labeled_vectors):

  if labeled_vectors == None or len(labeled_vectors) < 1:
    return vector

  max = -2
  most_similarity = None

  for v in labeled_vectors:

    distance = mu.cossim(vector, v)
    if distance > max:
      max = distance
      most_similarity = v

  return most_similarity
