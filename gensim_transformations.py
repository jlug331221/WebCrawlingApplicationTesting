import os
from gensim import corpora, models, matutils
import pyLDAvis.gensim

CURRENT_DIR = os.path.dirname(__file__)

'''
# Remove common (English) stop words from feature_vectors[key].
'''
def remove_stopwords(feature_vectors, key):
  feature_vectors_without_stop_words = []

  # common words to remove
  stop_list = set('your the is and or in be to of for not on with as by ay at this than '
                  'did an again 65 try you any no 10 alt a b c d e f g h i j k l m n o p'
                  'q r s t u v w x y z january february march april may june july august '
                  'september october november december com can invalid input but less '
                  'least long match characters consisting gender male female please '
                  'include new ctl00 content changed confirm later enter radio receivetext '
                  'receivetext1'.split())

  for feature_vector in feature_vectors.get(key):
    words_not_in_stopwords = []
    for word in feature_vector:
      if word not in stop_list:
        words_not_in_stopwords.append(word)
    feature_vectors_without_stop_words.append(words_not_in_stopwords)

  return feature_vectors_without_stop_words

'''
# Remove integers and empty spaces from feature_vectors[key].
'''
def remove_integers_and_empty_spaces_from_feature_vectors(feature_vectors, key):
  for feature_vector in feature_vectors.get(key):
    for token in feature_vector:
      if token.isdigit():
        feature_vector.remove(token)

      if token == ' ':
        feature_vector.remove(token)

'''
# Remove integers, empty spaces and stops words from feature_vectors.
'''
def pre_process_feature_vectors(feature_vectors):
  final_processed_feature_vectors = []

  for key in feature_vectors.keys():
    remove_integers_and_empty_spaces_from_feature_vectors(feature_vectors, key)

    pre_processed_feature_vectors = remove_stopwords(feature_vectors, key)

    for feature_vector in pre_processed_feature_vectors:
      final_processed_feature_vectors.append(feature_vector)

  return final_processed_feature_vectors

'''
# Apply bag of words transformation on feature_vectors and output transformation to
# '/gensim_transformation_output/bag_of_words.txt'. 
#
# Return bag of words transformations.
'''
def bag_of_words(feature_vectors):
  BoW = dict()

  with open(CURRENT_DIR + '/gensim_transformation_output/bag_of_words.txt', 'w') as BoW_output:
    for key in feature_vectors.keys():
      BoW[key] = []

      remove_integers_and_empty_spaces_from_feature_vectors(feature_vectors, key)

      preprocessed_feature_vectors = remove_stopwords(feature_vectors, key)

      dictionary = corpora.Dictionary(preprocessed_feature_vectors)

      # store the dictionary for later usage
      dictionary.save_as_text(CURRENT_DIR + '/gensim_transformation_output/dictionaries/' + key +
                              '.txt')

      BoW_output.write(key + ' has the following feature vectors:\n')

      for feature_vector in preprocessed_feature_vectors:
        BoW_output.write(str(feature_vector) + '\n')

      BoW_output.write('\nThe tokenized dictionary (unique words) is as follows:\n')
      BoW_output.write(str(dictionary.token2id) + '\n\n')

      BoW[key] = [dictionary.doc2bow(feature_vector)
                  for feature_vector in preprocessed_feature_vectors]

      # store corpus for later use
      corpora.MmCorpus.serialize(CURRENT_DIR + '/gensim_transformation_output/serialized_corpus/' +
                                 key + '.mm', BoW[key])

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

  with open(CURRENT_DIR + '/gensim_transformation_output/tfidf.txt', 'w') as tfidf_output:
    for key in bag_of_words.keys():
      tfidf[key] = None

      dictionary = corpora.Dictionary.load_from_text(CURRENT_DIR +
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
# Return LSA transformations.
'''
def LSA(tfidf):
  LSA = dict()

  with open(CURRENT_DIR + '/gensim_transformation_output/LSA.txt', 'w') as LSA_output:
    for key in tfidf.keys():
      LSA[key] = None

      dictionary = corpora.Dictionary.load_from_text(CURRENT_DIR +
                    '/gensim_transformation_output/dictionaries/' + key + '.txt')

      # initialize LSI model
      lsi_model = models.LsiModel(tfidf[key].corpus, num_topics=len(tfidf[key].corpus),
                                  id2word=dictionary)

      LSA_output.write(key + '\n')

      # use model to transform tf_idf -> LSA/LSI
      LSA[key] = lsi_model[tfidf[key].corpus]

      # Output LSA model topics
      LSA_output.write('\nLSI model topics:\n')
      for i in range(len(tfidf[key].corpus)):
        LSA_output.write(str(lsi_model.print_topic(i)) + '\n')
      # LSA_output.write(str(lsi_model.print_topics()))
      LSA_output.write('\n')

      LSA_output.write('\nLSI Model Projections:\n')
      LSA_output.write('S:\n' + str(lsi_model.projection.s) + '\n\n')
      LSA_output.write('U:\n' + str(LSA[key].obj.projection.u) + '\n')

      LSA_output.write('\nV^T is as follows:\n')
      LSA_output.write(str(matutils.corpus2dense(LSA[key], len(lsi_model.projection.s)).T /
                       lsi_model.projection.s))

      LSA_output.write('\n\n*************************************************************' + '\n\n')

  return LSA

'''
# Perform latent dirichlet allocation (LDA) transformation on tfidf. LDA is a vector dimensionality
# reduction transformation that is much like LSA/LSI, but with the addition of probabilistic
# distributions over topic words.
# 
# Utilizing pyLDAvis for the LDA model. The initial visualize variable is set to false. Set
# visualize to true to see a visual representation of the LDA model.
#
# Return LDA transformations.
'''
def LDA(tfidf, visualize=False):
  # Model used for output
  LDA_models = dict()

  # Main model that is updated with each training form; this model is used for similarity
  # inference and is saved to disk.
  LDA_model = None
  initialize_model = True

  with open(CURRENT_DIR + '/gensim_transformation_output/LDA.txt', 'w') as LDA_output:
    for key in tfidf.keys():
      LDA_models[key] = None

      dictionary = corpora.Dictionary.load_from_text(CURRENT_DIR +
                      '/gensim_transformation_output/dictionaries/' + key + '.txt')

      tokens2id = dictionary.token2id

      corpus = corpora.MmCorpus(CURRENT_DIR + '/gensim_transformation_output/serialized_corpus' +
                                key + '.mm')

      # initialize lda model
      lda = models.LdaModel(tfidf[key].corpus, num_topics=len(tfidf[key].corpus),
                            id2word=dictionary, passes=10)

      if initialize_model:
        LDA_model = models.LdaModel(tfidf[key].corpus, num_topics=len(tfidf[key].corpus),
                              id2word=dictionary, passes=10)
        initialize_model = False
      else:
        LDA_model.update(tfidf[key].corpus)

      LDA_output.write(key + '\n\n')

      LDA_models[key] = lda[tfidf[key].corpus]

      if visualize:
        # prepare LDA_topics model visualization
        LDA_vis_data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
        # save visualization
        pyLDAvis.save_html(LDA_vis_data, CURRENT_DIR + '/visualizations/LDA_topics/' + key)

      LDA_output.write('Corpus: ' + str(corpus) + '\n\n')

      LDA_output.write('LDA_model topics and their corresponding probabilities:\n')
      for i in range(len(tfidf[key].corpus)):
        LDA_output.write(str(lda.print_topic(i)) + '\n')

      LDA_output.write('\n** LDA_model top topic with probability: **\n')
      for i in range(len(tfidf[key].corpus)):
        top_topic_terms = lda.get_topic_terms(i, topn=1)

        for x in top_topic_terms:
          top_topic_terms = top_topic_terms,\
                            list(tokens2id.keys())[list(tokens2id.values()).index(x[0])]

        LDA_output.write(str(top_topic_terms) + '\n')

      LDA_output.write('\n\n*************************************************************' + '\n\n')
      lda = None

  return LDA_models

'''
# Build LDA model for training set data and store to disk. 
'''
def build_LDA_model_for_training_set(feature_vectors):
  if not os.path.exists(os.path.join(CURRENT_DIR, 'training_set_lda_model')):
    os.makedirs('training_set_lda_model')

  training_set_feature_vectors = pre_process_feature_vectors(feature_vectors)

  dictionary = corpora.Dictionary(training_set_feature_vectors)

  # store the dictionary for later usage
  dictionary.save_as_text(CURRENT_DIR + '/training_set_lda_model/training_set_dictionary.txt')

  corpus = [dictionary.doc2bow(feature_vector) for feature_vector in training_set_feature_vectors]

  # store corpus for later use
  corpora.MmCorpus.serialize(CURRENT_DIR + '/training_set_lda_model/training_corpus.mm', corpus)

  lda_model = models.LdaModel(corpus, num_topics=len(corpus), id2word=dictionary, passes=20)

  lda_model.save(CURRENT_DIR + '/training_set_lda_model/LDA.model')