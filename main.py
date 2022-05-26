import copy
import random
import time

import numpy
from typing import List


class Coordinate:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'Coordinate(x={self.x}, y={self.y})'


class Point(Coordinate):
    def get_distance(self, point: Coordinate):
        a = numpy.array((self.x, self.y, 0))
        b = numpy.array((point.x, point.y, 0))
        return numpy.linalg.norm(a - b)


class Customer:
    def __init__(self, consume: float, point: Point):
        self.consume = consume
        self.point = point


class F2ProblemDefinition:
    min_customers_success_tax = 0.95
    max_customer_to_point_distance = 85

    def __init__(self, customers: List[Customer], points: List[Point]):
        self.customers = customers
        self.points = points
        self.k = 1
        self.fitness = 0.0
        self.penal_fitness = 0.0
        self.penal = 0
        self.customer_point_distances = []
        self.solution = []

    @staticmethod
    def from_csv() -> 'F2ProblemDefinition':
        customers = []
        with open('clientes.csv') as file:
            content = file.readlines()
        for row in content:
            row = row.split(",")
            customers.append(Customer(point=Point(x=float(row[0]), y=float(row[1])), consume=float(row[2])))
        points = []
        for x in range(0, 1010, 10):
            for y in range(0, 1010, 10):
                points.append(Point(x=x, y=y))
        return F2ProblemDefinition(customers=customers, points=points)

    @staticmethod
    def random(n_customers=50, n_points=50) -> 'F2ProblemDefinition':
        random.seed(13)
        customers = [
            Customer(point=Point(x=random.randint(0, 1000), y=random.randint(0, 1000)), consume=random.randint(0, 100))
            for _ in range(n_customers)]
        points = [Point(x=random.random() * 2.0, y=random.random() * 2.0) for _ in range(n_points)]
        return F2ProblemDefinition(customers=customers, points=points)

    def objective_function(self) -> 'F2ProblemDefinition':
        total_distance = 0
        for index, custom_active_points in enumerate(self.solution):
            for active_index, active in enumerate(custom_active_points):
                if active:
                    distance = self.customer_point_distances[index][active_index]
                    total_distance = total_distance + distance
                    # capacity_consumed = 0
                    # for i in range(0, 600):
                    #     if self.solution[i][active_index]:
                    #         capacity_consumed = capacity_consumed + self.customers[i].consume
                    self.penal = self.penal + (distance - self.max_customer_to_point_distance) ** 3
        self.fitness = total_distance
        self.penal_fitness = self.fitness - self.penal
        return self

    def neighborhood_change(self, y: 'F2ProblemDefinition'):
        if y.penal_fitness < self.penal_fitness:
            y.k = 1
            return copy.deepcopy(y)
        else:
            self.k = self.k + 1
            return self

    def shake(self):
        y = copy.deepcopy(self)
        customer = random.randint(0, 600)
        if self.k == 1:
            index_max = numpy.argmax(y.solution[customer])
            y.solution[customer] = [p == min([index_max + 2, 10201]) for p in range(10201)]
        elif self.k == 2:
            index_max = numpy.argmax(y.solution[customer])
            y.solution[customer] = [p == min([index_max + 101, 10201]) for p in range(10201)]
        elif self.k == 3:
            index_max = numpy.argmax(y.solution[customer])
            y.solution[customer] = [p == min([index_max + 204, 10201]) for p in range(10201)]
        return y

    def get_initial_solution(self) -> 'F2ProblemDefinition':
        for customer_index, customer in enumerate(self.customers):
            customer_bool_solutions = []
            distances = [customer.point.get_distance(p) for p in self.points]
            self.customer_point_distances.append(distances)
            index_min = numpy.argmin(distances)
            for index, point in enumerate(self.points):
                if index == index_min:
                    customer_bool_solutions.append(True)
                else:
                    customer_bool_solutions.append(False)
            self.solution.append(customer_bool_solutions)
        return self


if __name__ == '__main__':
    problem = F2ProblemDefinition.from_csv()
    start = time.time()
    print("Starting to generate initial solution...")
    initial_solution = problem.get_initial_solution()
    kmax = 3
    num_evaluated_solutions = 0
    max_evaluated_solutions = 1000
    problem.objective_function()
    print(f'Initial Fitness: {problem.fitness}')
    while num_evaluated_solutions < max_evaluated_solutions:
        while problem.k <= kmax:
            new_solution = problem.shake()
            new_solution.objective_function()
            num_evaluated_solutions += 1
            problem = problem.neighborhood_change(y=new_solution)
            print(f'Fitness {num_evaluated_solutions}, k: {problem.k}: {problem.fitness}')
    # print(f"Time to generate initial solution: {time.time() - start} seconds")
    # total = F2ProblemDefinition.objective_function(solution=initial_solution)
    # print(f"Total distance initial solution: {total}")
