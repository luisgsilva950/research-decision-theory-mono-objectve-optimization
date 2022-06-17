from typing import Optional, List, Collection

import numpy
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

    def __eq__(self, other):
        return (self.x == other.x) and (self.y == other.y)

    def __lt__(self, other: 'Coordinate'):
        if self.x == other.x:
            return self.y < other.y
        return self.x < other.x

    def __gt__(self, other: 'Coordinate'):
        if self.x == other.x:
            return self.y > other.y
        return self.x > other.x

    def __ne__(self, other):
        return not (self == other)


class AccessPoint(Coordinate):

    def __init__(self, x: float, y: float, index: Optional[int]):
        super(AccessPoint, self).__init__(x, y)
        self.index = index

    def __hash__(self):
        return hash(self.index)

    def __eq__(self, other: 'AccessPoint'):
        return self.index == other.index

    def get_neighbor_indexes(self) -> List[int]:
        return [min(self.index + 1, 10200), self.index - 1, min(self.index + 1000, 10200), self.index - 1000]


class Customer:
    def __init__(self, consume: float, index: int, coordinates: Coordinate):
        self.consume = consume
        self.coordinates = coordinates
        self.index = index

    def get_closer_point(self, points: Collection[AccessPoint], distances: List[float] = None) -> AccessPoint:
        points = list(points)
        if not distances:
            distances = [self.coordinates.get_distance(p) for p in points]
        if len(points) != len(distances):
            distances = [self.coordinates.get_distance(p) for p in points]
        index_min = get_arg_min(distances)
        return points[index_min]
