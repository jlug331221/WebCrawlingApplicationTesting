import os, glob
from bs4 import BeautifulSoup as BS
from process_dom import extractFeatures

current_dir = os.path.dirname(__file__)
html_files_dir = current_dir + '/forms/'

featureVector = []

file = open(html_files_dir + 'simpleForm.html', 'r')
soupElement = BS(file, 'html.parser')

featureVector = extractFeatures(soupElement)

print(featureVector)
# for htmlFile in glob.glob(html_files_dir + '*.html'):
#   file = open(htmlFile, 'r')
#   soupElement = bs(file, 'html.parser')
#
#   featureVector += extractFeatures(soupElement)

