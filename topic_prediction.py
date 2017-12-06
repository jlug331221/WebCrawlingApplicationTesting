import os
from gensim import corpora, models
import numpy

CURRENT_DIR = os.path.dirname(__file__)
BASE_SAVED_LDA_MODEL_DIR = 'training_set_lda_model'

'''
# Return true if topic identifies feature_vector and false otherwise.
'''
def topic_predicted_correctly(topic, feature_vector):
  if topic == 'text':
    return False

  for word in feature_vector:
    if (topic == word) or (word in topic):
      return True

  return False

'''
# Predict the topic of features in forms_under_test_feature_vectors using an LDA model based
# on percentage_of_data of the training forms.
#
# The output is written to the lda_results directory with the current_iteration as the
# file identifier. The topic predictions and accuracies are saved accordingly.
'''
def predict_LDA(forms_under_test_feature_vectors, percentage_of_data, current_iteration):
  if not os.path.exists(os.path.join(CURRENT_DIR, 'lda_results/predictions')):
    os.makedirs('lda_results/predictions')

  if not os.path.exists(os.path.join(CURRENT_DIR, 'lda_results/accuracies')):
    os.makedirs('lda_results/accuracies')

  LDA_predicted_topics_output = open(CURRENT_DIR + '/lda_results/predictions/' +
                                     'Iteration(' + str(current_iteration) + ')_' +
                                     str(percentage_of_data) + '_LDA_predicted_topics_output.txt',
                                     'w')

  dictionary = corpora.Dictionary.load_from_text(BASE_SAVED_LDA_MODEL_DIR +
                                                 '/training_set_dictionary.txt')
  corpus = corpora.MmCorpus(BASE_SAVED_LDA_MODEL_DIR + '/training_corpus.mm')

  model_lda = models.LdaModel.load(BASE_SAVED_LDA_MODEL_DIR + '/LDA.model')

  LDA_predicted_topics_output.write('** Iteration: ' + str(current_iteration) + ' -> Using ' +
                                    str(percentage_of_data) + '% of training forms **\n')

  predicted_correctly = 0
  total_features = len(forms_under_test_feature_vectors)

  for feature_vector in forms_under_test_feature_vectors:
    # print('Feature vector under test: ' + str(feature_vector))
    LDA_predicted_topics_output.write('Feature vector under test: ' + str(feature_vector) + '\n')

    vec_bow = dictionary.doc2bow(feature_vector)
    vec_lda = model_lda[vec_bow]

    word_count_array = numpy.empty((len(vec_lda), 2), dtype=numpy.object) # create empty numpy array
    for i in range(len(vec_lda)): # populate with the results in 'vec_lda'
      word_count_array[i, 0] = vec_lda[i][0]
      word_count_array[i, 1] = vec_lda[i][1]

    idx = numpy.argsort(word_count_array[:, 1]) # sort based on probability
    idx = idx[::-1] # idx <- data with highest probability
    word_count_array = word_count_array[idx]

    topic = ''
    probability = 0.0
    if len(word_count_array) != 0:
      # store topic and probability in final
      final = model_lda.print_topic(word_count_array[0, 0], 1)

      topic = final.split('*')[1]
      probability = final.split('*')[0]

      if topic_predicted_correctly(topic, feature_vector):
        predicted_correctly += 1

    LDA_predicted_topics_output.write('Possible predicted topic: ' + str(topic) + '\n')
    LDA_predicted_topics_output.write('Probability: ' + str(probability) + '\n\n')

  LDA_accuracy_output = open(CURRENT_DIR + '/lda_results/accuracies/' +
                             'Iteration(' + str(current_iteration) + ')_' +
                             str(percentage_of_data) + '_LDA_accuracy.output.txt', 'w')

  prediction_accuracy = (predicted_correctly / total_features) * 100

  print('\tPredicted ' + str(predicted_correctly) + ' / ' + str(total_features)
        + ' feature vectors = %.2f%%' % prediction_accuracy + ' prediction accuracy')

  LDA_accuracy_output.write('Iteration ' + str(current_iteration) + ':\n' +
                            str(percentage_of_data) + '% of training forms used\n' +
                            'Prediction accuracy = %.2f%%' % prediction_accuracy)

  LDA_accuracy_output.close()
  LDA_predicted_topics_output.close()
