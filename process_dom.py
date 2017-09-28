import re

# Extract a feature vector from the input field of a form within an HTML file.
# Note: This is used for testing the extraction of a feature vector from a simple HTML
# file with a form. Form has one input element and one label. File used is in the 'forms'
# directory and is named 'simpleForm.html'.
def extractFeaturesSimple(soupElement):
  featureVector = []

  # Attribute list concerns input topic identification in an attribute list
  attributeList = ['id', 'name', 'value', 'type', 'placeholder', 'maxlength']

  inputElement = soupElement.find('input')

  labelFeatures = findClosestLabels(inputElement, iterations=5)
  if labelFeatures:
    featureVector += labelFeatures

  for key, value in inputElement.attrs.items():
    if value and key in attributeList:
      value = re.sub('[^a-zA-Z0-9]', ' ', value.lower())
      featureVector += [key, value]

  return featureVector

# Extract features from input elements in a form within an HTML file.
def extractFeatures(soupElement):
  featureVector = []
  inputElements = soupElement.find_all('input')
  for inputElement in inputElements:
    # Attribute list concerns input topic identification in an attribute list
    attributeList = ['id', 'name', 'value', 'type', 'placeholder', 'maxlength']

    labelFeatures = findClosestLabels(inputElement, iterations=5)
    if labelFeatures:
      featureVector += labelFeatures

    for key, value in inputElement.attrs.items():
      if value and key in attributeList:
        value = re.sub('[^a-zA-Z0-9]', ' ', value.lower())
        featureVector += [key, value]

  return featureVector

# Find the closest labels to a form input element in the DOM
def findClosestLabels(soupElement, iterations):
  if iterations == 0:
    return None

  siblings = []
  siblings += soupElement.find_previous_siblings()
  siblings += soupElement.find_next_siblings()

  labels = []
  candidateTags = ['span', 'label']
  for tag in candidateTags:
    labels += soupElement.find_previous_siblings(tag)
    labels += soupElement.find_next_siblings(tag)

    for sibling in siblings:
      labels += sibling.find_all(tag)

  if not labels:
    return findClosestLabels(soupElement, iterations - 1)
  else:
    content = []
    for label in labels:
      for ss in label.stripped_strings:
        content.append(re.sub('[^a-zA-Z0-9]', ' ', ss.lower()))

      if content:
        return content
      else:
        return findClosestLabels(soupElement.parent, iterations - 1)
