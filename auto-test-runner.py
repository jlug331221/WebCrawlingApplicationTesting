import os
import test_bank_fetcher as tb
import requests
import itertools

current_dir = os.path.dirname(__file__)
form_path = current_dir + '/forms/simple_form'

form_paths =[form_path]

# the function return a permutation of form_inputs' values
# input_tests: is an array of array of possible values for each input [["i1_val1", "in1_val2"], ["i2_val1", "i2_val2"]]
# return: all cases: [ ["i1_val1", "i2_val1"], ["i1_val1", "i2_val2"], ["i1_val2", "i2_val1"], ["i1_val2", "i2_val2"]]
# each element of the list is a test case.


def generate_test_cases(input_tests):

    return list(itertools.product(*input_tests))


def generate_cases_for_form(form_inputs, test_cases):

    form_number_inputs = len(form_inputs)
    my_cases = []
    for tc in test_cases:
        if form_number_inputs != len(tc):
            raise Exception("form inputs and number of values are not matched")

        my_input = {}
        for i in range(0, form_number_inputs):
            input_i = form_inputs[i]
            # set value for input_i
            for k in input_i:
                my_input[k] = tc[i]

        my_cases.append(my_input)

    return my_cases


print("ALL CASES TO TEST")

for my_form in form_paths:
    form_content = [{"firstName": ""}, {"last_name": '"'}]
    predicted_input_topics = ["first_name", "last_name"]

    # replace form value to test
    number_inputs = len(form_content)

    if number_inputs != len(predicted_input_topics):
        raise Exception('Incorrect number of predictions')

    possible_topics_tests = []
    for i in range(0, number_inputs):
        topic_i = predicted_input_topics[i]

        # get list of possible values for input_i
        tests = tb.fetch_bank_for_topic(topic_i)
        my_values = []
        valid_inputs = tests["valid"]
        invalid_inputs = tests["invalid"]
        my_values.extend(valid_inputs)
        my_values.extend(invalid_inputs)

        possible_topics_tests.append(my_values)

    my_test_cases = generate_test_cases(possible_topics_tests)
    form_cases = generate_cases_for_form(form_content, my_test_cases)

    print(form_cases)

