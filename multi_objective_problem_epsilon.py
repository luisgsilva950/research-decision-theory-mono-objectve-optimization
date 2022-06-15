import collections
import copy
from typing import List

import numpy

from graphic_plotter import GraphicPlotter
from models import ProblemDefinition, Customer, AccessPoint, Coordinate
from utils import column, get_points_distances_from_file, get_arg_min, get_arg_max


class EpsilonRestrictProblem(ProblemDefinition):
    min_customers_attended = 570
    max_distance = 85
    max_consumed_capacity = 150

    def __init__(self, customers: List[Customer], points: List[AccessPoint], customer_point_distances=None,
                 solution=None, active_points=None, penal: float = 0.0, penal_fitness: float = 0.0,
                 fitness: float = 0.0, k: int = 1, max_active_points: int = 100):
        self.customers = customers or []
        self.points = points or []
        self.k = k
        self.fitness = fitness
        self.penal_fitness = penal_fitness
        self.penal = penal
        self.customer_to_point_distances = customer_point_distances or []
        self.solution = solution or []
        self.active_points = active_points or []
        self.max_active_points = max_active_points

    @staticmethod
    def from_csv(max_active_points: int) -> 'EpsilonRestrictProblem':
        customers = []
        with open('clientes.csv') as file:
            content = file.readlines()
        for index, row in enumerate(content):
            row = row.split(",")
            customers.append(
                Customer(coordinates=Coordinate(x=float(row[0]), y=float(row[1])), consume=float(row[2]), index=index))
        points = []
        for x in range(0, 1010, 10):
            for y in range(0, 1010, 10):
                points.append(AccessPoint(x=x, y=y, index=len(points)))
        return EpsilonRestrictProblem(customers=customers, points=points,
                                      customer_point_distances=get_points_distances_from_file(),
                                      max_active_points=max_active_points)

    def objective_function(self) -> 'EpsilonRestrictProblem':
        total_distance = 0
        total_active_points = len(self.active_points)
        customers_attended_count = self.get_customers_attended_count()
        consumed_capacity_per_point = self.get_consumed_capacity()
        self.penal = 0.0
        penal_distance_count = 0
        penal_consumed_capacity_count = 0
        for customer_index, customer_active_points in enumerate(self.solution):
            for point_index, active in enumerate(customer_active_points):
                if active:
                    consumed_capacity = consumed_capacity_per_point[point_index]
                    distance = self.customer_to_point_distances[customer_index][point_index]
                    total_distance = total_distance + distance
                    if distance > self.max_distance:
                        self.penal = self.penal + (distance - self.max_distance)
                        penal_distance_count += 1
                    if consumed_capacity > self.max_consumed_capacity:
                        self.penal = self.penal + 2 * (consumed_capacity - self.max_consumed_capacity)
                        print(f"The consumed capacity restriction was outdated by customer: {customer_index}. "
                              f"Consumed capacity: {consumed_capacity}")
                        penal_consumed_capacity_count += 1
        if total_active_points > self.max_active_points:
            self.penal = self.penal + 400 * (total_active_points - self.max_active_points)
        if customers_attended_count < self.min_customers_attended:
            self.penal = self.penal + 600 * (self.min_customers_attended - customers_attended_count)
        self.fitness = total_distance
        self.penal_fitness = self.fitness + self.penal
        print(f"\033[3;94mThe distance restriction was counted as: {penal_distance_count}")
        print(f"\033[3;94mThe consumed capacity restriction was counted as: {penal_consumed_capacity_count}")
        print(f'\033[3;{"93m" if self.penal else "32m"}Solution with penal fitness: {self.penal_fitness}, '
              f'penal: {self.penal} total customers attended: {customers_attended_count} '
              f'and total active points: {total_active_points}')
        return self

    def deactivate_point(self, index: int):
        for customer in self.customers:
            self.solution[customer.index][index] = False

    def enable_customer_point(self, customer: Customer, point: AccessPoint):
        self.solution[customer.index][point.index] = True

    def disable_customer_point(self, customer: Customer, point: AccessPoint):
        self.solution[customer.index][point.index] = False

    def get_customers_attended_count(self) -> int:
        customers_attended_count = 0
        for customer_points in self.solution:
            customers_attended_count = customers_attended_count + max(customer_points)
        return customers_attended_count

    def get_consumed_capacity(self) -> dict:
        consumed_capacity_per_point = collections.defaultdict(float)
        for point in self.active_points:
            point_customers = column(self.solution, point.index)
            for customer_index, is_point_active_in_customer in enumerate(point_customers):
                if is_point_active_in_customer:
                    consumed_capacity_per_point[point.index] += self.customers[customer_index].consume
        return consumed_capacity_per_point

    def neighborhood_change(self, y: 'EpsilonRestrictProblem'):
        if y.penal_fitness < self.penal_fitness:
            y.k = 1
            y = EpsilonRestrictProblem(customers=y.customers, points=y.points,
                                       customer_point_distances=y.customer_to_point_distances,
                                       solution=copy.deepcopy(y.solution),
                                       active_points=copy.deepcopy(y.active_points), fitness=y.fitness, penal=y.penal,
                                       penal_fitness=y.penal_fitness,
                                       k=y.k)
            print(f"\033[3;94mCustomers attended: {y.get_customers_attended_count()} - "
                  f"Total active points: {len(y.active_points)}")
            return y

        else:
            self.k = self.k + 1
            print(f"\033[3;94mCustomers attended: {self.get_customers_attended_count()} - "
                  f"Total active points: {len(self.active_points)}")
            return self

    def deactivate_random_demand_point_and_connect_closer_point(self):
        random_point: AccessPoint = numpy.random.choice(list(self.active_points))
        active_indexes: List[int] = [p.index for p in self.active_points]
        possible_indexes: List[int] = random_point.get_neighbor_indexes()
        possible_indexes: List[int] = [i for i in possible_indexes if i not in active_indexes]
        for customer in self.customers:
            if self.solution[customer.index][random_point.index]:
                possible_distances = [self.customer_to_point_distances[customer.index][i] for i in possible_indexes]
                closer_index = possible_indexes[get_arg_min(possible_distances)]
                self.enable_customer_point(customer=customer, point=self.points[closer_index])
        self.deactivate_point(index=random_point.index)

    def connect_random_customers_to_closer_active_demand_point(self, size: int = 5):
        random_customers: List[Customer] = list(numpy.random.choice(self.customers, size=size))
        for customer in random_customers:
            index_max = get_arg_max(self.solution[customer.index])
            closer_point = customer.get_closer_point(points=self.active_points,
                                                     distances=self.customer_to_point_distances[customer.index])
            if self.solution[customer.index][index_max] and closer_point.index != index_max:
                self.enable_customer_point(customer=customer, point=closer_point)
                self.disable_customer_point(customer=customer, point=self.points[index_max])

    def deactivate_random_customers(self, size: int = 5):
        random_customers: List[Customer] = list(
            numpy.random.choice([c for c in self.customers if max(self.solution[c.index]) > 0], size=size))
        for customer in random_customers:
            index_max = get_arg_max(self.solution[customer.index])
            self.disable_customer_point(customer=customer, point=self.points[index_max])

    def enable_random_customers(self, size: int = 5):
        random_customers: List[Customer] = list(
            numpy.random.choice([c for c in self.customers if max(self.solution[c.index]) == 0], size=size))
        for customer in random_customers:
            closer_point = customer.get_closer_point(points=self.active_points)
            self.enable_customer_point(customer=customer, point=closer_point)

    def shake_k1(self):
        self.connect_random_customers_to_closer_active_demand_point()

    def shake_k2(self):
        self.deactivate_random_customers()
        self.enable_random_customers()

    def shake_k3(self):
        self.deactivate_random_demand_point_and_connect_closer_point()
        self.deactivate_random_demand_point_and_connect_closer_point()

    def update_active_points(self):
        self.active_points = set()
        for i in range(10201):
            is_active_point = max(column(self.solution, i))
            if is_active_point:
                self.active_points.add(self.points[i])

    def shake(self):
        y = EpsilonRestrictProblem(customers=self.customers, points=self.points,
                                   customer_point_distances=self.customer_to_point_distances,
                                   solution=copy.deepcopy(self.solution),
                                   active_points=copy.deepcopy(self.active_points), fitness=self.fitness,
                                   penal=self.penal,
                                   penal_fitness=self.penal_fitness,
                                   k=self.k)
        if self.k == 1:
            y.shake_k1()
        elif self.k == 2:
            y.shake_k2()
        elif self.k == 3:
            y.shake_k3()
        y.update_active_points()
        return y

    def get_points_with_space_100(self) -> List[AccessPoint]:
        points = []
        for p in self.points:
            if p.x % 100 == 0 and p.y % 100 == 0 and p.x >= 100 and p.y >= 100:
                points.append(p)
        return points

    def plot_solution(self):
        plotter = GraphicPlotter(title='Connexions', connexions=self.get_connexions())
        plotter.plot()

    def get_connexions(self):
        result = list()
        for point in self.active_points:
            point_customers = []
            for customer in self.customers:
                if self.solution[customer.index][point.index]:
                    point_customers.append(customer.coordinates)
            result.append((point, point_customers))
        return result

    def get_initial_solution(self) -> 'EpsilonRestrictProblem':
        all_points = self.get_points_with_space_100()
        self.active_points = []
        self.solution = []
        for customer in self.customers:
            customer_bool_solutions = []
            distances = [self.customer_to_point_distances[customer.index][p.index] for p in all_points]
            index = get_arg_min(distances)
            closer_point = all_points[index]
            if distances[index] > self.max_distance and len(self.active_points) < self.max_active_points:
                closer_point = customer.get_closer_point(points=self.points,
                                                         distances=self.customer_to_point_distances[customer.index])
            if closer_point.index not in [p.index for p in self.active_points]:
                self.active_points.add(closer_point)
            for point_index, point in enumerate(self.points):
                customer_bool_solutions.append(point_index == closer_point.index)
            self.solution.append(customer_bool_solutions)
        self.update_active_points()
        self.objective_function()
        print(f"\033[3;92mTotal active points on initial solution: {len(self.active_points)}, "
              f"initial penal: {self.penal}")
        return self
