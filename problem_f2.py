from typing import List

from sklearn.cluster import KMeans

from models import Customer, AccessPoint, Coordinate
from problem_definition import ProblemDefinition
from rvns import Rvns
from utils import get_points_distances_from_file, get_arg_min


class ProblemDefinitionF2(ProblemDefinition):
    min_customers_attended = 570
    max_distance = 85
    max_active_points = 100
    max_consumed_capacity = 150

    def __init__(self, customers: List[Customer], points: List[AccessPoint], customer_point_distances=None,
                 solution=None, active_points=None, penal: float = 0.0, penal_fitness: float = 0.0,
                 fitness: float = 0.0, k: int = 1, total_distance: float = 0):
        self.customers = customers or []
        self.points = points or []
        self.k = k
        self.fitness = fitness
        self.penal_fitness = penal_fitness
        self.penal = penal
        self.customer_to_point_distances = customer_point_distances or []
        self.solution = solution or []
        self.active_points = active_points or set()
        self.total_distance = total_distance

    @staticmethod
    def from_csv() -> 'ProblemDefinitionF2':
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
        return ProblemDefinitionF2(customers=customers, points=points,
                                   customer_point_distances=get_points_distances_from_file())

    def objective_function(self) -> 'ProblemDefinitionF2':
        total_distance = 0
        total_active_points = len(self.active_points)
        customers_attended_count = self.get_customers_attended_count()
        consumed_capacity_per_point = self.get_consumed_capacity()
        self.penal = 0.0
        penal_distance_count = 0
        penal_consumed_capacity_count = 0
        for active_point in self.active_points:
            for customer in self.customers:
                point_index = active_point.index
                customer_index = customer.index
                active = self.solution[customer.index][active_point.index]
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

    def neighborhood_change(self, y: 'ProblemDefinitionF2'):
        if y.penal_fitness < self.penal_fitness:
            y.k = 1
            y = ProblemDefinitionF2(customers=y.customers.copy(), points=y.points.copy(),
                                    customer_point_distances=y.customer_to_point_distances,
                                    solution=[p.copy() for p in y.solution],
                                    active_points=y.active_points.copy(), fitness=y.fitness, penal=y.penal,
                                    penal_fitness=y.penal_fitness,
                                    k=y.k, total_distance=y.total_distance)
            print(f"\033[3;94mCustomers attended: {y.get_customers_attended_count()} - "
                  f"Total active points: {len(y.active_points)}")
            return y

        else:
            self.k = self.k + 1
            print(f"\033[3;94mCustomers attended: {self.get_customers_attended_count()} - "
                  f"Total active points: {len(self.active_points)}")
            return self

    def shake_k1(self):
        self.connect_random_customers_to_closer_active_access_point()

    def shake_k2(self):
        self.deactivate_random_customers()
        self.enable_random_customers()

    def shake_k3(self):
        self.deactivate_less_demanded_access_point()
        self.enable_random_customers(size=1, points=self.points)

    def shake(self):
        y = ProblemDefinitionF2(customers=self.customers.copy(), points=self.points.copy(),
                                customer_point_distances=self.customer_to_point_distances,
                                solution=[p.copy() for p in self.solution],
                                active_points=self.active_points.copy(), fitness=self.fitness,
                                penal=self.penal,
                                penal_fitness=self.penal_fitness,
                                k=self.k, total_distance=self.total_distance)
        if self.k == 1:
            y.shake_k1()
        elif self.k == 2:
            y.shake_k2()
        elif self.k == 3:
            y.shake_k3()
        y.update_active_points()
        return y


if __name__ == '__main__':
    problem_f2 = ProblemDefinitionF2.from_csv()
    rvns_f2 = Rvns(problem=problem_f2, max_solutions_evaluations=300, n=5)
    rvns_f2.run()
