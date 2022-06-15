import numpy.random

from multi_objective_problem_pondered_sum import PonderedSumProblem
from problem_f1 import ProblemDefinitionF1
from problem_f2 import ProblemDefinitionF2
from rvns import Rvns
from utils import save_file, load_file
from matplotlib import pyplot as plt

N_EVALUATIONS = 100
N_CURVES = 5
N_WEIGHTS = 30

# def plot_result():
#     non_viable_f1 = []
#     non_viable_f2 = []
#     curve = None
#     for curve_number in range(2):
#         min_index = curve_number * 30
#         max_index = (curve_number + 1) * 30
#         curve = load_file(f"execution_{curve_number}_{N_EVALUATIONS}_{N_CURVES}_{N_WEIGHTS}_pondered_sum.json")
#         viable_solution = curve["is_eligible_solution"]
#         f1_values = curve["f1_values"]
#         f2_values = curve["f2_values"]
#         eligible_f1_values = [f1 for index, f1 in enumerate(f1_values) if viable_solution[min_index:max_index]]
#         eligible_f2_values = [f2 for index, f2 in enumerate(f2_values) if viable_solution[min_index:max_index]]
#         non_viable_f1.extend([f1 for index, f1 in enumerate(f1_values) if not viable_solution[min_index:max_index]])
#         non_viable_f2.extend([f2 for index, f2 in enumerate(f2_values) if not viable_solution[min_index:max_index]])
#         plt.plot(eligible_f1_values, eligible_f2_values, '.', label=f'Execution {curve_number}')
#     plt.plot(curve["f1_values"][:2], curve["f2_values"][:2], 'bs', label=f'Mono optimization')
#     plt.plot(non_viable_f1, non_viable_f2, 'rx', label=f'Non viable solutions')
#     plt.legend()
#     plt.show()
#     plt.savefig(fname='pondered_sum_f1.png')
#     plt.close()


if __name__ == '__main__':
    is_eligible_solution = []
    problem_f1 = ProblemDefinitionF1.from_csv()
    rvns_f1 = Rvns(problem=problem_f1, max_solutions_evaluations=N_EVALUATIONS, n=1)
    rvns_f1.run()
    problem_f2 = ProblemDefinitionF2.from_csv()
    rvns_f2 = Rvns(problem=problem_f2, max_solutions_evaluations=N_EVALUATIONS, n=1)
    rvns_f2.run()
    f1_min_solution = rvns_f1.best_solution
    f2_min_solution = rvns_f2.best_solution
    plt.xlabel("f1(x)")
    plt.ylabel("f2(x)")
    for iteration in range(N_CURVES):
        is_eligible_solution = [f1_min_solution.penal == 0, f2_min_solution.penal == 0]
        f1_values = [f1_min_solution.penal_fitness, len(f2_min_solution.active_points)]
        f2_values = [f1_min_solution.total_distance, f2_min_solution.penal_fitness]
        for _ in range(N_WEIGHTS):
            w1 = numpy.random.uniform(0, 1)
            problem = PonderedSumProblem.from_csv(w1=w1, w2=1 - w1)
            rvns = Rvns(problem=problem, max_solutions_evaluations=N_EVALUATIONS, n=1)
            rvns.run()
            f1_values.append(len(rvns.best_solution.active_points))
            f2_values.append(rvns.best_solution.total_distance)
            is_eligible_solution.append(rvns.best_solution.penal == 0)
        save_file(file_name=f"execution_{iteration}_{N_EVALUATIONS}_{N_CURVES}_{N_WEIGHTS}_pondered_sum.json",
                  f1_values=f1_values, f2_values=f2_values, is_eligible_solution=is_eligible_solution)
