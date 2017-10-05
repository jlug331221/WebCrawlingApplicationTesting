import re

#
# Extract features from input elements in a form within the DOM (HTML file).
#
def extractFeatures(inputElement):
  featureVector = []

  inputElementAttrs = inputElement.attrs.items()
  # Only extract features for DOM input elements that are not hidden and that are not submit
  #  buttons. These are the ones that are going to be tested.
  if not containsHiddenValueAttr(inputElementAttrs):
    # Attribute list concerns input topic identification in an attribute list
    attributeList = ['id', 'name', 'value', 'type', 'placeholder', 'maxlength']

    labelFeatures = findClosestLabels(inputElement, iterations=5)
    if labelFeatures:
      featureVector += labelFeatures

    for key, value in inputElementAttrs:
      if value and key in attributeList:
        value = re.sub('[^a-zA-Z0-9]', ' ', value.lower())
        featureVector += [key, value]

  return featureVector

#
# Find the closest labels to a form input element in the DOM (HTML file)
#
def findClosestLabels(inputElement, iterations):
  if iterations == 0:
    return None

  siblings = []
  siblings += inputElement.find_previous_siblings()
  siblings += inputElement.find_next_siblings()

  labels = []
  candidateTags = ['span', 'label']
  for tag in candidateTags:
    labels += inputElement.find_previous_siblings(tag)
    labels += inputElement.find_next_siblings(tag)

    for sibling in siblings:
      labels += sibling.find_all(tag)

  if not labels:
    return findClosestLabels(inputElement, iterations - 1)
  else:
    content = []
    for label in labels:
      for ss in label.stripped_strings:
        content.append(re.sub('[^a-zA-Z0-9]', ' ', ss.lower()))

      if content:
        return content
      else:
        return findClosestLabels(inputElement.parent, iterations - 1)

#
# Returns true if inputElementAttrs contains a value of 'hidden' or 'submit' and false
# otherwise
#
def containsHiddenValueAttr(inputElementAttrs):
  for key, value in inputElementAttrs:
    strValue = ''.join(value)
    if strValue.lower() == 'hidden' or strValue.lower() == 'submit':
      return True

  return False