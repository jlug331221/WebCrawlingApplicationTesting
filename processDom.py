import re


# Extract a feature vector from the input fields of a form within an HTML file
def extractFeatures(soupElement):
  featureVector = []

  # Attribute list concerns input topic identification in an attribute list
  attributeList = ['id', 'name', 'value', 'type', 'placeholder', 'maxlength']

  labelFeatures = findClosestLabels(soupElement, iterations=5)
  if labelFeatures:
    featureVector += labelFeatures

  for key, value in soupElement.attrs.items():
    if key in attributeList:
      featureVector += [key, re.sub('[^a-zA-Z0-9]', ' ', value.lower())]


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
    labels += soupElement.find_previous_siblings(tag)

    for sibling in siblings:
      labels += sibling.find_all(tag)

  if not labels:
    return findClosestLabels(soupElement, iterations - 1)
  else:
    content = []
    for label in labels:
      for ss in label.stripped_strings:
        content.append(re.sub('[^a-zA-Z0-9]', ' ', s.lower()))

      if content:
        return content
      else:
        return findClosestLabels(soupElement.parent, iterations - 1)
