import json

import numpy
from typing import List


def column(matrix, i):
    return [row[i] for row in matrix]


def get_arg_max(_list) -> int:
    return numpy.argmax(_list)


def get_arg_min(_list) -> int:
    return numpy.argmin(_list)


def get_points_distances_from_file() -> List[List[float]]:
    with open('customer_point_distances.json', 'r') as f:
        value = json.load(f)
    return value
