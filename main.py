import os, glob
from bs4 import BeautifulSoup as BS
from process_dom import extractFeatures

current_dir = os.path.dirname(__file__)
html_files_dir = current_dir + '/forms/'

# Extract feature vectors from all college forms in the 'forms' directory
featureVectors = dict()
for htmlFile in glob.glob(html_files_dir + '*.html'):
  file = open(htmlFile, 'r')
  file_name = htmlFile.split('forms', 1)[1]
  featureVectors[file_name] = []

  soupElement = BS(file, 'html.parser')

  # Find form elements in DOM file (could be more than one form)
  formElements = soupElement.find_all('form')
  for formElement in formElements:
    inputElements = formElement.find_all('input')
    for inputElement in inputElements:
      featureVector = extractFeatures(inputElement)
      if featureVector:
        featureVectors[file_name].append(featureVector)
        #featureVectors.append(featureVector)

  print('\nFile \'', file_name, '\' has the following extracted feature vectors:\n')
  for featureVector in featureVectors[file_name]:
    print(featureVector)
  print('\n************************************************')
