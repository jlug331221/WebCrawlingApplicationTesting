import os, glob
from bs4 import BeautifulSoup as BS
from process_dom import extractFeatures, extractFeaturesSimple

current_dir = os.path.dirname(__file__)
html_files_dir = current_dir + '/forms/'

# Extract feature vector from the simple form within the 'forms' directory
featureVector = []
file = open(html_files_dir + 'simpleForm.html', 'r')
soupElement = BS(file, 'html.parser')
featureVector = extractFeaturesSimple(soupElement)
print('Simple form feature vector: ', featureVector, '\n')

# Extract feature vectors from all college forms in the 'forms' directory
featureVectors = []
for htmlFile in glob.glob(html_files_dir + '*.html'):
  file = open(htmlFile, 'r')

  #Find form elements in DOM file (could be more than one form)
  soupElement = BS(file, 'html.parser')
  formElements = soupElement.find_all('form')
  for formElement in formElements:
    featureVectors.append(extractFeatures(formElement))

for featureVector in featureVectors:
  print(featureVector)