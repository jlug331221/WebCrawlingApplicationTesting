import os, glob
from bs4 import BeautifulSoup as bs
from processDOM import extractFeatures

current_dir = os.path.dirname(__file__)
html_files_dir = current_dir + '/forms/'

for htmlFile in glob.glob(html_files_dir + '*.html'):
  file = open(htmlFile, 'r')
  soupElement = bs(file, 'html.parser')

  extractFeatures(soupElement)

