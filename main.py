import os, glob
from bs4 import BeautifulSoup as BS
from process_dom import extract_features
import vector_transformations as vt
import gensim_transformations as gt

current_dir = os.path.dirname(__file__)
html_files_dir = current_dir + '/forms/'

'''
#
# Print the contents of feature_vectors.
#
'''
def print_feature_vectors(feature_vectors):
  for html_file in glob.glob(html_files_dir + '*.html'):
    file_name = html_file.split('forms', 1)[1]

    print('\nFile \'', file_name, '\' has the following extracted feature vectors:\n')
    for feature_vector in feature_vectors[file_name]:
      print(feature_vector)
    print('\n************************************************')

'''
#
# Extract feature vectors from the 'forms' directory. This directory contains
# all of the static university forms for testing purposes.
#
'''
def extract_feature_vectors_from_university_forms():
  # Extract feature vectors from all university forms in the 'forms' directory
  feature_vectors = dict()
  for html_file in glob.glob(html_files_dir + '*.html'):
    file = open(html_file, 'r')
    file_name = html_file.split('forms', 1)[1]
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
#
# Main global procedure.
#
'''
def main():
  feature_vectors = extract_feature_vectors_from_university_forms()

  # bag_of_words = vt.bag_of_words_transformation(feature_vectors)
  #
  # tfidf = vt.tfidf_transformation(bag_of_words)
  #
  # LSA = vt.LSA_transformation(tfidf)

'''
#
# Automatically extract features when executing this module.
#
'''
if __name__ == '__main__':
  main()