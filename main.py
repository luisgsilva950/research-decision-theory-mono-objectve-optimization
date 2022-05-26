import random
import numpy
from typing import List


class Coordinate:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class Point(Coordinate):
    def get_distance(self, point: Coordinate):
        a = numpy.array((self.x, self.y, 0))
        b = numpy.array((point.x, point.y, 0))
        return numpy.linalg.norm(a - b)


class Customer:
    def __init__(self, consume: int, point: Point):
        self.consume = consume
        self.point = point


class F2ProblemDefinition:
    min_customers_success_tax = 0.95
    max_customer_to_point_distance = 85

    def __init__(self, customers: List[Customer], points: List[Point]):
        self.customers = customers
        self.points = points

    @staticmethod
    def from_csvs(csv_customers_file_name: str, csv_points_file_name: str):
        ...


def get_f2_initial_solution(problem_definition: F2ProblemDefinition) -> List[List[float]]:
    solution = []
    for customer in problem_definition.customers:
        customer_solution = []
        random_point_index = random.randint(0, len(problem_definition.points))
        for index, point in enumerate(problem_definition.points):
            if index == random_point_index:
                customer_solution.append(customer.point.get_distance(point))
            else:
                customer_solution.append(0.0)
        solution.append(customer_solution)
    return solution


def get_f2_problem_definition(max_customer_to_point_distance=85, min_customers_success_tax=0.95):
    random.seed(13)
    # customers = [Customer(point=Point(x=random.random() * 2.0, y=random.random() * 2.0), consume=random.randint(0, 100))
    #              for _ in
    #              range(n_customers)]
    # points = [Point(x=random.random() * 2.0, y=random.random() * 2.0) for _ in range(n_points)]
    customers = []
    points = []
    problem_definition = F2ProblemDefinition(customers=customers, points=points)
    return problem_definition


if __name__ == '__main__':
    problem = get_f2_problem_definition()
    initial_solution = get_f2_initial_solution(problem_definition=problem)
    print(initial_solution[0])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
