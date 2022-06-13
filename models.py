from typing import Optional, List
import matplotlib.pyplot as plt

import numpy
import numpy as np

from utils import get_arg_min


class Coordinate:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def get_distance(self, point: 'Coordinate'):
        a = numpy.array((self.x, self.y, 0))
        b = numpy.array((point.x, point.y, 0))
        return numpy.linalg.norm(a - b)

    def __repr__(self):
        return f'Coordinate(x={self.x}, y={self.y})'


class AccessPoint(Coordinate):

    def __init__(self, x: float, y: float, index: Optional[int]):
        super(AccessPoint, self).__init__(x, y)
        self.index = index

    def get_neighbor_indexes(self) -> List[int]:
        return [min(self.index + 1, 10200), self.index - 1, min(self.index + 1000, 10200), self.index - 1000]


class Customer:
    def __init__(self, consume: float, index: int, coordinates: Coordinate):
        self.consume = consume
        self.coordinates = coordinates
        self.index = index

    def get_closer_point(self, points: List[AccessPoint], distances: List[float] = None) -> AccessPoint:
        if not distances:
            distances = [self.coordinates.get_distance(p) for p in points]
        if len(points) != len(distances):
            distances = [self.coordinates.get_distance(p) for p in points]
        index_min = get_arg_min(distances)
        return points[index_min]


class ProblemDefinition:
    k: int
    penal_fitness: float
    fitness: float
    penal: float
    active_points: List[AccessPoint]
    total_distance: float = 0

    def objective_function(self) -> 'ProblemDefinition':
        ...

    def neighborhood_change(self, y: 'ProblemDefinition') -> 'ProblemDefinition':
        ...

    def shake(self) -> 'ProblemDefinition':
        ...

    def get_initial_solution(self) -> 'ProblemDefinition':
        ...

    def plot_solution(self) -> None:
        ...
