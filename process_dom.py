import re

'''
#
# Extract features from input_element in a form within the DOM.
#
'''
def extract_features(input_element):
  feature_vector = []

  input_element_attrs = input_element.attrs.items()
  # Only extract features for DOM input elements that are not hidden and that are not submit
  #  buttons. These are the ones that are going to be tested.
  if not contains_hidden_valueAttr_or_submitButton(input_element_attrs):
    # Attribute list concerns input topic identification in an attribute list
    attr_list = ['id', 'name', 'value', 'type', 'placeholder', 'maxlength']

    label_features = find_closest_labels(input_element)
    if label_features:
      feature_vector += label_features

    for key, value in input_element_attrs:
      if value and key in attr_list:
        value = re.sub('[^a-zA-Z0-9]', ' ', value.lower())
        feature_vector += [key, value]

  return feature_vector

'''
#
# Find the closest labels to input_element in the DOM.
#
# Default iterations is set to 5.
#
'''
def find_closest_labels(input_element, iterations=5):
  if iterations == 0:
    return None

  siblings = []
  siblings += input_element.find_previous_siblings()
  siblings += input_element.find_next_siblings()

  labels = []
  candidateTags = ['span', 'label']
  for tag in candidateTags:
    labels += input_element.find_previous_siblings(tag)
    labels += input_element.find_next_siblings(tag)

    for sibling in siblings:
      labels += sibling.find_all(tag)

  if not labels:
    return find_closest_labels(input_element, iterations - 1)
  else:
    content = []
    for label in labels:
      for ss in label.stripped_strings:
        content.append(re.sub('[^a-zA-Z0-9]', ' ', ss.lower()))

      if content:
        return content
      else:
        return find_closest_labels(input_element.parent, iterations - 1)

'''
#
# Returns true if input_element_attrs contains a value of 'hidden' or 'submit'
# and false otherwise.
#
'''
def contains_hidden_valueAttr_or_submitButton(input_element_attrs):
  for key, value in input_element_attrs:
    strValue = ''.join(value)
    if strValue.lower() == 'hidden' or strValue.lower() == 'submit':
      return True

  return False