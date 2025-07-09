import numpy as np


def prediction_correctness(actual, predicted, small_error=10):

    difference = _percent_difference(actual, predicted)

    prediction_type = np.array([categorize_difference(x, small_error=small_error) for x in difference])

    return prediction_type


def _percent_difference(actual, predicted):
    difference = 100 * (predicted - actual) / actual

    return difference


def categorize_difference(x, small_error=10):
    type = "Correct"

    if x > small_error:
        type = "Overestimate"
    elif x < -small_error:
        type = "Underestimate"

    return type