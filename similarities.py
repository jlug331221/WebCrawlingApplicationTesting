import os
from gensim import corpora, models
import numpy

CURRENT_DIR = os.path.dirname(__file__)
BASE_SAVED_LDA_MODEL_DIR = 'training_set_lda_model'

def predict_LDA(forms_under_test_feature_vectors):
  if not os.path.exists(os.path.join(CURRENT_DIR, 'lda_results')):
    os.makedirs('lda_results')

  dictionary = corpora.Dictionary.load_from_text(BASE_SAVED_LDA_MODEL_DIR +
                                                 '/training_set_dictionary.txt')
  corpus = corpora.MmCorpus(BASE_SAVED_LDA_MODEL_DIR + '/training_corpus.mm')

  model_lda = models.LdaModel.load(BASE_SAVED_LDA_MODEL_DIR + '/LDA.model')

  for feature_vector in forms_under_test_feature_vectors:
    print('Feature vector under test: ' + str(feature_vector))

    vec_bow = dictionary.doc2bow(feature_vector)
    vec_lda = model_lda[vec_bow]

    word_count_array = numpy.empty((len(vec_lda), 2), dtype=numpy.object) # create empty numpy array
    for i in range(len(vec_lda)): # populate with the results in 'vec_lda'
      word_count_array[i, 0] = vec_lda[i][0]
      word_count_array[i, 1] = vec_lda[i][1]

    idx = numpy.argsort(word_count_array[:, 1]) # sort based on probability
    idx = idx[::-1] # idx <- data with highest probability
    word_count_array = word_count_array[idx]

    final = model_lda.print_topic(word_count_array[0, 0], 1) # store topic and probability in final

    topic = final.split('*')[1]
    probability = final.split('*')[0]

    print('Possible predicted topic: ' + str(topic))
    print('Predicted probability: ' + str(probability) + '\n')
