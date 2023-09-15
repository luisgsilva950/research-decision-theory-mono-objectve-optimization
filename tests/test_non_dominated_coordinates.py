from unittest import TestCase

from models import Coordinate, CoordinatesCalculator


class TestParetoFrontierAlgorithm(TestCase):

    def test_get_pareto_great_solutions(self):
        points = [Coordinate(x=0, y=1), Coordinate(x=0, y=2), Coordinate(x=0, y=3), Coordinate(x=0, y=4),
                  Coordinate(x=1, y=1), Coordinate(x=1, y=2), Coordinate(x=1, y=3), Coordinate(x=1, y=4),
                  Coordinate(x=2, y=1)]

        result = CoordinatesCalculator.find_pareto_frontier(points=points)

        self.assertEqual(1, len(result))
        self.assertEqual(Coordinate(x=0, y=1), result[0])

    def test_get_pareto_great_solutions_2(self):
        points = [Coordinate(x=2, y=2), Coordinate(x=3, y=3), Coordinate(x=4, y=4), Coordinate(x=5, y=5),
                  Coordinate(x=6, y=6)]

        result = CoordinatesCalculator.find_pareto_frontier(points=points)

        self.assertEqual(1, len(result))
        self.assertEqual(Coordinate(x=2, y=2), result[0])

    def test_get_pareto_great_solutions_3(self):
        points = [Coordinate(x=6, y=1), Coordinate(x=1, y=6), Coordinate(x=5, y=2), Coordinate(x=2, y=5),
                  Coordinate(x=3, y=4), Coordinate(x=4, y=3), Coordinate(x=3.5, y=3.5),
                  Coordinate(x=3.5, y=3), Coordinate(x=3, y=3.5)]

        result = CoordinatesCalculator.find_pareto_frontier(points=points)

        self.assertEqual(6, len(result))
        self.assertEqual(Coordinate(x=1, y=6), result[0])
        self.assertEqual(Coordinate(x=2, y=5), result[1])
        self.assertEqual(Coordinate(x=3, y=3.5), result[2])
        self.assertEqual(Coordinate(x=3.5, y=3), result[3])
        self.assertEqual(Coordinate(x=5, y=2), result[4])
        self.assertEqual(Coordinate(x=6, y=1), result[5])
