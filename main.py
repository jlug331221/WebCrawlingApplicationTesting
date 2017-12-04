import os, glob
from bs4 import BeautifulSoup as BS
from process_dom import extract_features
import vector_transformations as vt
import gensim_transformations as gt
import random as rand
from math import ceil

current_dir = os.path.dirname(__file__)
html_files_dir = current_dir + '/training_forms/'

'''
# Print the contents of feature_vectors.
'''
def print_feature_vectors(feature_vectors):
  for html_file in glob.glob(html_files_dir + '*.html'):
    file_name = html_file.split('training_forms', 1)[1]

    print('\nFile \'', file_name, '\' has the following extracted feature vectors:\n')
    for feature_vector in feature_vectors[file_name]:
      print(feature_vector)
    print('\n************************************************')

'''
# Extract feature vectors from the 'training_forms' directory. This directory contains
# all of the static university training_forms for testing purposes.
'''
def extract_feature_vectors_from_university_forms():
  # Extract feature vectors from all university training_forms in the 'training_forms' directory
  feature_vectors = dict()
  for html_file in glob.glob(html_files_dir + '*.html'):
    file = open(html_file, 'r')
    file_name = html_file.split('training_forms', 1)[1]
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
def perform_sklean_transformations(feature_vectors):
  bag_of_words = vt.bag_of_words_transformation(feature_vectors)

  tfidf = vt.tfidf_transformation(bag_of_words)

  LSA = vt.LSA_transformation(tfidf)

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

  for html_file in glob.glob(html_files_dir + '*.html'):
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
# Main global procedure.
'''
def main():
  # feature_vectors = extract_feature_vectors_from_university_forms()

  # sklearn transformations
  # perform_sklean_transformations(feature_vectors)

  # gensim transformations for testing purposes
  # perform_gensim_transformations(feature_vectors)

  training_forms = randomly_pick_training_forms(training_data_percentage=60)

  # gt.build_LDA_model_for_entire_training_set(feature_vectors)

  print('\nDone')

'''
# Automatically extract features when executing this module.
'''
if __name__ == '__main__':
  main()