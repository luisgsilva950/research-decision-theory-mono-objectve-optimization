import collections
import copy
import random
import time

import numpy
from typing import List
import json


def column(matrix, i):
    return [row[i] for row in matrix]


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
    min_customers_attended = 570
    max_customer_to_point_distance = 85
    max_active_points = 25
    max_consumed_capacity = 150

    def __init__(self, customers: List[Customer], points: List[Point]):
        self.customers = customers
        self.points = points
        self.k = 1
        self.fitness = 0.0
        self.penal_fitness = 0.0
        self.penal = 0
        self.customer_point_distances = []
        self.points_status = [False for _ in range(10201)]
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
        total_active_points = sum(self.points_status)
        customers_attended_count = self.get_customers_attended_count()
        consumed_capacity_per_point = self.get_consumed_capacity()
        for customer_index, customer_active_points in enumerate(self.solution):
            for point_index, active in enumerate(customer_active_points):
                if active:
                    consumed_capacity = consumed_capacity_per_point[point_index]
                    distance = self.customer_point_distances[customer_index][point_index]
                    total_distance = total_distance + distance
                    if distance > self.max_customer_to_point_distance:
                        self.penal = self.penal + (distance - self.max_customer_to_point_distance) ** 3
                    if consumed_capacity > self.max_consumed_capacity:
                        self.penal = self.penal + (consumed_capacity - self.max_consumed_capacity) ** 3
        if total_active_points > self.max_active_points:
            self.penal = self.penal + (total_active_points - self.max_active_points) ** 3
        print(f"Customers attended: {customers_attended_count} - Total active points: {total_active_points}")
        if customers_attended_count < self.min_customers_attended:
            self.penal = self.penal + (self.min_customers_attended - customers_attended_count) ** 3
        self.fitness = total_distance
        self.penal_fitness = self.fitness + self.penal
        return self

    def get_customers_attended_count(self) -> int:
        customers_attended_count = 0
        for customer_points in self.solution:
            customers_attended_count = customers_attended_count + max(customer_points)
        return customers_attended_count

    def get_consumed_capacity(self) -> dict:
        consumed_capacity_per_point = collections.defaultdict(float)
        for point_index, is_used_point in enumerate(self.points_status):
            if is_used_point:
                point_customers = column(self.solution, point_index)
                for customer_index, is_point_active_in_customer in enumerate(point_customers):
                    if is_point_active_in_customer:
                        consumed_capacity_per_point[point_index] += self.customers[customer_index].consume
        return consumed_capacity_per_point

    def neighborhood_change(self, y: 'F2ProblemDefinition'):
        if y.penal_fitness < self.penal_fitness:
            y.k = 1
            return copy.deepcopy(y)
        else:
            self.k = self.k + 1
            return self

    def shake(self):
        y = copy.deepcopy(self)
        if self.k == 1:
            random_point = random.randint(0, 10201)
            for customer in range(600):
                points = y.solution[customer]
                points[random_point] = False
                y.solution[customer] = points
        elif self.k == 2:
            for _ in range(5):
                customer = random.randint(0, 600)
                y.solution[customer] = [False for _ in range(10201)]
        elif self.k == 3:
            for _ in range(5):
                customer = random.randint(0, 600)
                index_max = numpy.argmax(y.solution[customer])
                y.solution[customer] = [p == min([index_max + 2, 10201]) for p in range(10201)]
        for i in range(10201):
            self.points_status[i] = max(column(self.solution, i))
        return y

    def save_points_distances_on_file(self):
        with open('customer_point_distances.json', 'w') as f:
            json.dump(self.customer_point_distances, f)

    def get_points_distances_from_file(self):
        with open('customer_point_distances.json', 'r') as f:
            value = json.load(f)
        return value or []

    def get_initial_solution(self) -> 'F2ProblemDefinition':
        for customer_index, customer in enumerate(self.customers):
            customer_bool_solutions = []
            self.customer_point_distances = self.get_points_distances_from_file()
            if self.customer_point_distances:
                distances = self.customer_point_distances[customer_index]
            else:
                distances = [customer.point.get_distance(p) for p in self.points]
                self.customer_point_distances.append(distances)
            index_min = numpy.argmin(distances)
            for index, point in enumerate(self.points):
                if index == index_min:
                    customer_bool_solutions.append(True)
                else:
                    customer_bool_solutions.append(False)
            self.solution.append(customer_bool_solutions)
        self.save_points_distances_on_file()
        for i in range(10201):
            self.points_status[i] = max(column(self.solution, i))
        print(f"Total active points on initial solution: {sum(self.points_status)}")
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
