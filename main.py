import os, glob
from bs4 import BeautifulSoup as BS
from process_dom import extract_features
import sklearn_transformations as sklt
import gensim_transformations as gt
import topic_prediction
import random as rand
import shutil
from math import ceil

CURRENT_DIR = os.path.dirname(__file__)
TRAINING_FORMS_DIR = CURRENT_DIR + '/training_forms/'
FORMS_UNDER_TEST_DIR = CURRENT_DIR + '/forms_under_test/'
TOTAL_ITERATIONS = 10

'''
# Remove training set LDA model from disk.
'''
def cleanup_training_set_lda_model_directory():
  print('Performing cleanup...\n')
  shutil.rmtree('training_set_lda_model')
  print('Cleanup complete')

'''
# Print the contents of feature_vectors in dir_path/forms_type.
'''
def print_feature_vectors(feature_vectors, dir_path, forms_type):
  for html_file in glob.glob(dir_path + '*.html'):
    file_name = html_file.split(forms_type, 1)[1]

    print('\nFile \'', file_name, '\' has the following extracted feature vectors:\n')
    for feature_vector in feature_vectors[file_name]:
      print(feature_vector)
    print('\n************************************************')

'''
# Extract feature vectors as specified in forms_list from dir.
'''
def extract_feature_vectors(dir, forms_list):
  # Extract feature vectors from all university training_forms in the 'training_forms' directory
  feature_vectors = dict()
  for html_file in glob.glob(dir + '*.html'):
    file = open(html_file, 'r')

    if dir == TRAINING_FORMS_DIR:
      file_name = html_file.split('training_forms', 1)[1]
    if dir == FORMS_UNDER_TEST_DIR:
      file_name = html_file.split('forms_under_test', 1)[1]

    if file_name in forms_list:
      feature_vectors[file_name] = []

      soup_element = BS(file, 'html.parser')

      # Find form elements in DOM file (could be more than one form)
      form_elements = soup_element.find_all('form')
      for form_element in form_elements:
        form_input_elements = form_element.find_all('input')
        for form_input_element in form_input_elements:
          feature_vector = extract_features(form_input_element)
          if feature_vector:
            feature_vectors[file_name].append(feature_vector)

  return feature_vectors

'''
# Perform sklearn transformations.
'''
def perform_sklearn_transformations(feature_vectors):
  bag_of_words = sklt.bag_of_words_transformation(feature_vectors)

  tfidf = sklt.tfidf_transformation(bag_of_words)

  LSA = sklt.LSA_transformation(tfidf)

'''
# Perform gensim transformations.
'''
def perform_gensim_transformations(feature_vectors):
  bag_of_words = gt.bag_of_words(feature_vectors)

  tfidf = gt.tf_idf(bag_of_words)

  LSA = gt.LSA(tfidf)

  LDA_models = gt.LDA(tfidf)

'''
# Randomly pick training_data_percentage of forms in the directory 'training_forms' to build
# an LDA training model.
'''
def randomly_pick_training_forms(training_data_percentage):
  total_training_forms = []
  training_forms = []

  for html_file in glob.glob(TRAINING_FORMS_DIR + '*.html'):
    file_name = html_file.split('training_forms', 1)[1]
    total_training_forms.append(file_name)

  # Choose training_data_percentage of the training forms
  for i in range(0, int(ceil((training_data_percentage/100) * len(total_training_forms)))):
    choice = rand.choice(total_training_forms)
    while choice in training_forms:
      choice = rand.choice(total_training_forms)
    training_forms.append(choice)

  return training_forms

'''
# Returns all form names from the 'forms_under_test' directory.
'''
def get_forms_under_test():
  forms_under_test = []

  for html_file in glob.glob(FORMS_UNDER_TEST_DIR + '*.html'):
    file_name = html_file.split('forms_under_test', 1)[1]
    forms_under_test.append(file_name)

  return forms_under_test

'''
# Main global procedure.
'''
def main():
  training_data_percentages = [25, 50, 75, 90]

  current_iteration = 1

  print('Getting forms under test...')
  forms_under_test = get_forms_under_test()
  forms_under_test_feature_vectors = extract_feature_vectors(FORMS_UNDER_TEST_DIR, forms_under_test)
  print('Pre-processing forms under test...\n')
  forms_under_test_feature_vectors = \
    gt.pre_process_feature_vectors(forms_under_test_feature_vectors)

  while current_iteration <= TOTAL_ITERATIONS:
    for i in range(len(training_data_percentages)):
      training_forms = randomly_pick_training_forms(
        training_data_percentage=training_data_percentages[i])

      training_feature_vectors = extract_feature_vectors(TRAINING_FORMS_DIR, training_forms)

      print('Iteration ' + str(current_iteration) + ' -> ', end='')
      print('Generating LDA model with ' + str(training_data_percentages[i]) +
            '% of the training forms...')
      gt.build_LDA_model_for_training_set(training_feature_vectors)

      topic_prediction.predict_LDA(forms_under_test_feature_vectors, training_data_percentages[i],
                                   current_iteration)
    print('Iteration ' + str(current_iteration) + ' complete.\n')

    current_iteration += 1

  cleanup_training_set_lda_model_directory()

  print('\nDone')

'''
# Automatically extract features when executing this module.
'''
if __name__ == '__main__':
  main()