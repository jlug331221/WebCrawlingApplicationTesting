import os, glob
from bs4 import BeautifulSoup as BS
from process_dom import extractFeatures

current_dir = os.path.dirname(__file__)
html_files_dir = current_dir + '/forms/'

# Extract feature vectors from all college forms in the 'forms' directory
featureVectors = []
for htmlFile in glob.glob(html_files_dir + '*.html'):
  file = open(htmlFile, 'r')
  soupElement = BS(file, 'html.parser')

  # Find form elements in DOM file (could be more than one form)
  formElements = soupElement.find_all('form')
  for formElement in formElements:
    inputElements = formElement.find_all('input')
    for inputElement in inputElements:
      featureVector = extractFeatures(inputElement)
      if featureVector:
        featureVectors.append(featureVector)

  print('\nFile \'', htmlFile, '\' has the following extracted feature vectors:\n')
  for featureVector in featureVectors:
    print(featureVector)
  print('\n************************************************')
