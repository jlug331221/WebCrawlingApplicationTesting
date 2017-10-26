from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD

'''
#
# Return bag of words transformation on feature_vectors.
# 
# bag of words := count of significant words in each feature vector.
#
# Each bag of words transformation produces a document-term matrix
# rows    := length of the list being transformed
# columns := unique words in each feature vector
# 
# Example:
#
  ['first name', 'id', 'firstname', 'name', 'firstname', 'maxlength', '45', 'type', 'text']
  [  [0 1 0 0 0 1 0 0]
     [0 0 0 1 0 0 0 0]
     [0 0 1 0 0 0 0 0]
     [0 0 0 0 0 1 0 0]
     [0 0 1 0 0 0 0 0]
     [0 0 0 0 1 0 0 0]
     [1 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 1]
     [0 0 0 0 0 0 1 0] ]
  Feature names: ['45', 'first', 'firstname', 'id', 'maxlength', 'name', 'text', 'type']
#
# 9 x 8 document-term matrix
# 
# The first sub-list := [0 1 0 0 0 1 0 0]
# says that 'first' and 'name' appear one time in the document.
# This is verified by printing the feature names of the vectorizer, which is the last list in the
# example.
#
'''
def bag_of_words_transformation(feature_vectors):
  bag_of_words = []
  i = 0

  count_vectorizer = CountVectorizer()
  for key in feature_vectors.keys():
    for feature_vector in feature_vectors.get(key):
      print(feature_vector)
      bag_of_words.append(count_vectorizer.fit_transform(feature_vector).toarray())
      print(bag_of_words[i])
      print('Feature names: ' + str(count_vectorizer.get_feature_names()))
      print('\n')
      i = i + 1

  return bag_of_words

'''
#
# Apply and return the tfidf transformation on bag_of_words. The transformation converts
# the counts in bag_of_words to real-value weights (real numbers).
#
'''
def tfidf_transformation(bag_of_words):
  tfidf = []
  i = 0

  tfidf_transformer = TfidfTransformer(smooth_idf=False)

  for word_count in bag_of_words:
    tfidf.append(tfidf_transformer.fit_transform(word_count).toarray())
    print(tfidf[i])
    i = i + 1
    print('\n')

  return tfidf

'''
#
# Apply and return the latent semantic analysis (LSA) transformation on tfidf. This 
# transformation reduces the dimension of the vector space by means of singular value
# decomposition (SVD).
#
'''
def LSA_transformation(tfidf):
  LSA = []
  i = 0

  svd = TruncatedSVD()

  for i_doc_freq in tfidf:
    LSA.append(svd.fit_transform(i_doc_freq))
    print(svd.explained_variance_ratio_)
    print(svd.explained_variance_)
    i = i + 1

  return LSA
