import datetime
import random

import numpy

from models import CoordinatesCalculator, Coordinate
from multi_objective_problem_epsilon import EpsilonRestrictProblem
from problem_f1 import ProblemDefinitionF1
from problem_f2 import ProblemDefinitionF2
from rvns import Rvns
from utils import load_file, save_file, get_arg_max
from matplotlib import pyplot as plt

N_EVALUATIONS = 300
N_CURVES = 1


def plot_result(scope: str, n=N_EVALUATIONS, n_curves=N_CURVES):
    non_viable_f1 = []
    non_viable_f2 = []
    all_points = []
    curve = None
    plt.xlabel("F1")
    plt.ylabel("F2")
    for curve_number in range(N_CURVES):
        curve = load_file(f"{scope}_results/execution_{curve_number}_{n}_{n_curves}_{scope}.json")
        viable_solution = curve["is_eligible_solution"]
        f1s = curve["f1_values"]
        f2s = curve["f2_values"]
        eligible_f1s = [f1 for index, f1 in enumerate(f1s) if viable_solution[index]]
        eligible_f2s = [f2 for index, f2 in enumerate(f2s) if viable_solution[index]]
        non_viable_f1.extend([f1 for index, f1 in enumerate(f1s) if not viable_solution[index]])
        non_viable_f2.extend([f2 for index, f2 in enumerate(f2s) if not viable_solution[index]])
        all_points.extend([Coordinate(x=x, y=eligible_f2s[index]) for index, x in enumerate(eligible_f1s)])
        plt.plot(eligible_f1s, eligible_f2s, '.', label=f'Execution {curve_number}')
    pareto_greats = CoordinatesCalculator.get_non_dominated_coordinates(points=all_points)
    plt.plot([p.x for p in pareto_greats], [p.y for p in pareto_greats], 'k^', label=f'Pareto greats')
    plt.plot(curve["f1_values"][:2], curve["f2_values"][:2], 'bs', label=f'Mono optimization')
    plt.plot(non_viable_f1, non_viable_f2, 'rx', label=f'Non viable solutions')
    plt.legend()
    plt.savefig(fname=f'{scope}_results/{scope}_{n}_{n_curves}_pareto_great.png')
    plt.close()


if __name__ == '__main__':
    # plot_result(scope="pondered_sum")
    for iteration in range(N_CURVES):
        is_eligible_solution = [True, True]
        f1_values = [47.0, 100]
        f2_values = [26351.853168212587, 11663.26]
        consumed_capacity_solution = []
        max_distance_solution = []
        solution = []
        for max_active_points in range(47, 100, 1):
            problem = EpsilonRestrictProblem.from_csv(max_active_points=max_active_points)
            rvns = Rvns(problem=problem, max_solutions_evaluations=N_EVALUATIONS, n=1, kmax=5,
                        n_clusters=max_active_points)
            rvns.run()
            f1_values.append(len(rvns.best_solution.active_points))
            f2_values.append(rvns.best_solution.total_distance)
            is_eligible_solution.append(rvns.best_solution.penal == 0)
            # solution_indexes = dict((c_index, get_arg_max(c)) for c_index, c in enumerate(rvns.best_solution.solution))
            solution.append(rvns.best_solution.solution)
            max_distance_solution.append(rvns.best_solution.get_max_customer_distance())
            consumed_capacity_solution.append(rvns.best_solution.get_max_consumed_capacity())
        save_file(
            file_name=f"epsilon_restrict_results/execution_{iteration}_{N_EVALUATIONS}_{N_CURVES}_epsilon_restrict_{random.randint(0, 200000)}.json",
            f1_values=f1_values, f2_values=f2_values, is_eligible_solution=is_eligible_solution,
            consumed_capacity=consumed_capacity_solution,
            distance=max_distance_solution, solution=solution)
