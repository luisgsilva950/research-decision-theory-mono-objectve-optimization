from multi_objective_problem_epsilon import EpsilonRestrictProblem
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
    f1_values = [f1_min_solution.penal_fitness]
    f2_values = [f2_min_solution.penal_fitness]
    for max_active_points in range(int(f1_min_solution.fitness), 100, 1):
        problem = EpsilonRestrictProblem.from_csv(max_active_points=max_active_points)
        rvns = Rvns(problem=problem, max_solutions_evaluations=150)
        rvns.run()
        f1_values.append(len(rvns.best_solution.active_points))
        f2_values.append(rvns.best_solution.penal_fitness)
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.plot(f1_values, f2_values, 'o')
    plt.legend()
    plt.savefig(fname='epsilon_restrict_f1.png')
    plt.close()
    save_points_epsilon_restrict_problem(f1_values=f1_values, f2_values=f2_values)
