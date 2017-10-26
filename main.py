import os, glob
from bs4 import BeautifulSoup as BS
from process_dom import extract_features
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD

current_dir = os.path.dirname(__file__)
html_files_dir = current_dir + '/forms/'

def extract_and_print_feature_vectors_for_static_forms():
  # Extract feature vectors from all college forms in the 'forms' directory
  feature_vectors = dict()
  for htmlFile in glob.glob(html_files_dir + '*.html'):
    file = open(htmlFile, 'r')
    file_name = htmlFile.split('forms', 1)[1]
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

    print('\nFile \'', file_name, '\' has the following extracted feature vectors:\n')
    for feature_vector in feature_vectors[file_name]:
      print(feature_vector)
    print('\n************************************************')

  return feature_vectors

def main():
  feature_vectors = extract_and_print_feature_vectors_for_static_forms()

  # Perform bag of words on simpleform.html
  bag_of_words = None
  vectorizer = CountVectorizer()
  print('\n')
  for key in feature_vectors.keys():
    if key == '\simpleForm.html':
      for feature_vector in feature_vectors.get(key):
        print(feature_vector)
        bag_of_words = vectorizer.fit_transform(feature_vector).toarray()
        print(bag_of_words)
        print(vectorizer.get_feature_names())
        # print(vectorizer.inverse_transform(bag_of_words[1]))
        print('\n')

  # Once bag of words model is fit on feature vector extracted from simpleForm.html,
  # transform counts to real numbers using tfidf
  transformer = TfidfTransformer(smooth_idf=False)
  tf_idf = transformer.fit_transform(bag_of_words).toarray()

  print(tf_idf)

  print('\n')

  # Apply LSI/LSA (Single Value Decomposition)
  # LSA = Latent Semantic Analysis
  svd = TruncatedSVD(n_iter=5)
  svd.fit_transform(tf_idf)
  print(svd.explained_variance_ratio_)
  print(svd.explained_variance_)

if __name__ == '__main__':
  main()