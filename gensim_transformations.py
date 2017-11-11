import os
from gensim import corpora, models, similarities, matutils

current_dir = os.path.dirname(__file__)

'''
# Apply bag of words transformation on feature_vectors and output transformation to
# '/gensim_transformation_output/bag_of_words.txt'. 
#
# Return bag of words transformation.
'''
def bag_of_words(feature_vectors):
  BoW = dict()

  with open(current_dir + '/gensim_transformation_output/bag_of_words.txt', 'w') as BoW_output:
    for key in feature_vectors.keys():
      BoW[key] = []

      dictionary = corpora.Dictionary(feature_vectors.get(key))
      # store the dictionary for later usage in LSI transformation
      dictionary.save_as_text(current_dir + '/gensim_transformation_output/dictionaries/' + key +
                              '.txt')

      BoW_output.write(key + ' has the following feature vectors:\n')

      for feature_vector in feature_vectors.get(key):
        BoW_output.write(str(feature_vector) + '\n')

      BoW_output.write('\nThe tokenized dictionary (unique words) is as follows:\n')
      BoW_output.write(str(dictionary.token2id) + '\n\n')

      BoW[key] = [dictionary.doc2bow(feature_vector) for feature_vector in feature_vectors.get(key)]

      BoW_output.write('Bag of words counts for each feature vector:\n')
      for BoW_vect in BoW[key]:
        BoW_output.write(str(BoW_vect) + '\n')

      BoW_output.write('\n*************************************************************' + '\n\n')

  return BoW

'''
# Apply term frequency inverse document frequency transformation on bag_of_words and
# output transformation to '/gensim_transformation_output/tfidf.txt'.
#
# Return tf_idf transformation.
'''
def tf_idf(bag_of_words):
  tfidf = dict()

  with open(current_dir + '/gensim_transformation_output/tfidf.txt', 'w') as tfidf_output:
    for key in bag_of_words.keys():
      tfidf[key] = None

      dictionary = corpora.Dictionary.load_from_text(current_dir +
                    '/gensim_transformation_output/dictionaries/' + key + '.txt')

      # initialize TF_IDF model
      tfidf_model = models.TfidfModel(bag_of_words[key], id2word=dictionary)

      tfidf_output.write(key + '\n')

      # use model to transform BoW vectors to tfidf vectors
      tfidf[key] = tfidf_model[bag_of_words[key]]

      for tfidf_vect in tfidf[key]:
        tfidf_output.write(str(tfidf_vect) + '\n')

      tfidf_output.write('\n*************************************************************' + '\n\n')

  return tfidf

'''
# Apply latent semantic analysis for dimensionality reduction on tfidf and output transformation
# results to '/gensim_transformation_output/LSA.txt'.
#
# Return LSA transformation.
'''
def LSA(tfidf):
  LSA = dict()

  with open(current_dir + '/gensim_transformation_output/LSA.txt', 'w') as LSA_output:
    for key in tfidf.keys():
      LSA[key] = None

      dictionary = corpora.Dictionary.load_from_text(current_dir +
                    '/gensim_transformation_output/dictionaries/' + key + '.txt')

      # initialize LSI model
      lsi_model = models.LsiModel(tfidf[key], num_topics=len(tfidf[key]), id2word=dictionary)

      LSA_output.write(key + '\n')

      # use model to transform tf_idf -> LSA/LSI
      LSA[key] = lsi_model[tfidf[key]]

      # Output most important topics
      LSA_output.write('\nLSI model topics:\n')
      LSA_output.write(str(lsi_model.print_topics()))
      LSA_output.write('\n')

      LSA_output.write('\nLSI Model Projections:\n')
      LSA_output.write('S:\n' + str(lsi_model.projection.s) + '\n\n')
      LSA_output.write('U:\n' + str(LSA[key].obj.projection.u) + '\n')

      LSA_output.write('\nV^T is as follows:\n')
      LSA_output.write(str(matutils.corpus2dense(LSA[key], len(lsi_model.projection.s)).T /
                       lsi_model.projection.s))

      LSA_output.write('\n\n*************************************************************' + '\n\n')

  return LSA
