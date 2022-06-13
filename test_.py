import numpy.random
from typing import List

from multi_objective_problem_pondered_sum import PonderedSumProblem
from problem_f1 import ProblemDefinitionF1
from problem_f2 import ProblemDefinitionF2
from rvns import Rvns
from utils import save_points_epsilon_restrict_problem
from matplotlib import pyplot as plt
from multiprocessing import Pool, Process
from concurrent.futures import ThreadPoolExecutor
from typing import Callable


def log_exceptions(func: Callable[[dict, dict], None]):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            print(f"Error when processing function {func.__name__}")
            raise err
    return wrapper


def execute_async(fn, args):
    wrapper = log_exceptions(fn)
    with ThreadPoolExecutor(max_workers=25) as executor:
        return list(executor.map(lambda _args: wrapper(*_args), args))


def _run_rvns(*args, **kwargs):
    w1 = numpy.random.uniform(0, 1)
    problem = PonderedSumProblem.from_csv(w1=w1, w2=1 - w1)
    rvns = Rvns(problem=problem, max_solutions_evaluations=2, n=1)
    rvns.run()
    return rvns.best_solution


def generate_file_for_execution(iteration: int, f1_initial_values: List[float], f2_initial_values: List[float]):
    f1_values = f1_initial_values.copy()
    f2_values = f2_initial_values.copy()
    args = [(10,) for _ in range(25)]
    results = execute_async(_run_rvns, args)
    for result in results:
        f1_values.append(len(result.active_points))
        f2_values.append(result.total_distance)
    plt.plot(f1_values, f2_values, '.', label=f'Execution {iteration}')
    save_points_epsilon_restrict_problem(file_name=f"execution_{iteration}_pondered_sum.json", f1_values=f1_values,
                                         f2_values=f2_values)


if __name__ == '__main__':
    problem_f1 = ProblemDefinitionF1.from_csv()
    rvns_f1 = Rvns(problem=problem_f1, max_solutions_evaluations=2, n=1)
    rvns_f1.run()
    problem_f2 = ProblemDefinitionF2.from_csv()
    rvns_f2 = Rvns(problem=problem_f2, max_solutions_evaluations=2, n=1)
    rvns_f2.run()
    f1_min_solution = rvns_f1.best_solution
    f2_min_solution = rvns_f2.best_solution
    f1_values = [f1_min_solution.penal_fitness, len(f2_min_solution.active_points)]
    f2_values = [f1_min_solution.total_distance, f2_min_solution.penal_fitness]
    pool = Pool(100)
    processes = [pool.apply_async(generate_file_for_execution, args=(iteration, f1_values, f2_values)) for iteration in range(5)]
    result = [p.get() for p in processes]

    # plt.xlabel("f1(x)")
    # plt.ylabel("f2(x)")
    # for iteration in range(5):
    #     for _ in range(25):
    #         w1 = numpy.random.uniform(0, 1)
    #         problem = PonderedSumProblem.from_csv(w1=w1, w2=1 - w1)
    #         rvns = Rvns(problem=problem, max_solutions_evaluations=150, n=1)
    #         rvns.run()
    #         f1_values.append(len(rvns.best_solution.active_points))
    #         f2_values.append(rvns.best_solution.total_distance)
    #     plt.plot(f1_values, f2_values, '.', label=f'Execution {iteration}')
    #     save_points_epsilon_restrict_problem(file_name=f"execution_{iteration}_pondered_sum.json", f1_values=f1_values,
    #                                          f2_values=f2_values)
    # plt.legend()
    # plt.show()
    # plt.savefig(fname='pondered_sum_f1.png')
    # plt.close()
