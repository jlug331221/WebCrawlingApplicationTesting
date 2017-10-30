import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD

current_dir = os.path.dirname(__file__)

'''
#
# Return bag of words transformation on feature_vectors (for each document).
#
# Output is written to 'transformation_output/bag_of_words.txt'.
# 
# bag of words := count of significant words in each feature vector.
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
# The first sub-list := [0 1 0 0 0 1 0 0] says that 'first' and 'name' appear one time in the
# document.
# This is verified by printing the feature names of the vectorizer, which is the last list in the
# example.
#
'''
def bag_of_words_transformation(feature_vectors):
  bag_of_words = dict()
  count_vectorizer = CountVectorizer()

  with open(current_dir + '/transformation_output/bag_of_words.txt', 'w') as BoW_output:
    for key in feature_vectors.keys():
      bag_of_words[key] = []
      BoW_output.write(key + '\n')

      feature_vector_words = []
      vector_word_string = ''
      for feature_vector in feature_vectors.get(key):
        BoW_output.write(str(feature_vector) + '\n')
        for feature_vector_word in feature_vector:
          vector_word_string += str(feature_vector_word) + ' '
        feature_vector_words.append(vector_word_string)

      bag_of_words[key] = count_vectorizer.fit_transform(feature_vector_words)

      BoW_output.write(str(bag_of_words[key]) + '\n')
      # BoW_output.write(str(bag_of_words[key].toarray()) + '\n')
      BoW_output.write(str(bag_of_words[key].shape) + '\n')
      BoW_output.write('Feature names: ' + str(count_vectorizer.get_feature_names()) + '\n\n')
      BoW_output.write('*************************************************************' + '\n\n')

  return bag_of_words

'''
#
# Apply and return the TF-IDF transformation on bag_of_words. The transformation converts
# the counts in bag_of_words to real-value weights (real numbers). TF-IDF measures the
# relevance of the word and not the frequency.
# TF-IDF first measures the number of times a word appears in a document. The inverse
# document frequency aspect handles words such as 'and' or 'but' which appear in all
# documents and those words are given less relevance (weight). The result of TF-IDF is
# words that are frequent and distinctive.
#
'''
def tfidf_transformation(bag_of_words):
  tfidf = dict()
  tfidf_transformer = TfidfTransformer()

  with open(current_dir + '/transformation_output/tfidf.txt', 'w') as tfidf_output:
    for key in bag_of_words.keys():
      tfidf[key] = []
      tfidf_output.write(key + '\n')

      tfidf[key] = tfidf_transformer.fit_transform(bag_of_words[key])

      tfidf_output.write(str(tfidf[key]) + '\n\n')

    tfidf_output.write('*************************************************************' + '\n\n')

  return tfidf

'''
#
# Apply and return the latent semantic analysis (LSA) transformation on tfidf. This 
# transformation reduces the dimension of the vector space by means of singular value
# decomposition (SVD).
#
'''
def LSA_transformation(tfidf):
  LSA = dict()
  svd = TruncatedSVD()

  with open(current_dir + '/transformation_output/LSA.txt', 'w') as LSA_output:
    for key in tfidf.keys():
      LSA[key] = []
      LSA_output.write(key + '\n')

      LSA[key] = svd.fit_transform(tfidf[key])

      LSA_output.write(str(list(enumerate(LSA[key]))) + '\n\n')
      LSA_output.write('V^T: ' + str(svd.components_) + '\n\n')

      LSA_output.write('*************************************************************' + '\n\n')

  return LSA
