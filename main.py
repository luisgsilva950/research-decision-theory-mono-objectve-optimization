import numpy.random

from multi_objective_problem_epsilon import EpsilonRestrictProblem
from multi_objective_problem_pondered_sum import PonderedSumProblem
from problem_f1 import ProblemDefinitionF1
from problem_f2 import ProblemDefinitionF2
from rvns import Rvns
from utils import save_points_epsilon_restrict_problem
from matplotlib import pyplot as plt

if __name__ == '__main__':
    problem_f1 = ProblemDefinitionF1.from_csv()
    rvns_f1 = Rvns(problem=problem_f1, max_solutions_evaluations=150, n=1)
    rvns_f1.run()
    problem_f2 = ProblemDefinitionF2.from_csv()
    rvns_f2 = Rvns(problem=problem_f2, max_solutions_evaluations=150, n=1)
    rvns_f2.run()
    f1_min_solution = rvns_f1.best_solution
    f2_min_solution = rvns_f2.best_solution
    for _ in range(5):
        f1_values = [f1_min_solution.penal_fitness]
        f2_values = [f2_min_solution.penal_fitness]
        for _ in range(20):
            w1 = numpy.random.uniform(0, 1)
            problem = PonderedSumProblem.from_csv(w1=w1, w2=1 - w1)
            rvns = Rvns(problem=problem, max_solutions_evaluations=150, n=1)
            rvns.run()
            f1_values.append(len(rvns.best_solution.active_points))
            f2_values.append(rvns.best_solution.penal_fitness)
        plt.xlabel("f1(x)")
        plt.ylabel("f2(x)")
        plt.plot(f1_values, f2_values, 'o', label=f'Execution {_}')
        save_points_epsilon_restrict_problem(file_name=f"execution_{_}_pondered_sum.json", f1_values=f1_values,
                                             f2_values=f2_values)
    plt.legend()
    plt.savefig(fname='pondered_sum_f1.png')
    plt.close()
