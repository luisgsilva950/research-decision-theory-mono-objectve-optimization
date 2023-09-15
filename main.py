from models import CoordinatesCalculator, Coordinate
from utils import load_file
from matplotlib import pyplot as plt

N_EVALUATIONS = 250
N_CURVES = 5
N_WEIGHTS = 30


def plot_result(scope: str, n=N_EVALUATIONS, n_curves=N_CURVES, n_weights=N_WEIGHTS):
    non_viable_f1 = []
    non_viable_f2 = []
    all_points = []
    curve = None
    plt.xlabel("F1")
    plt.ylabel("F2")
    for curve_number in range(N_CURVES):
        curve = load_file(f"{scope}_results/execution_{curve_number}_{n}_{n_curves}_{n_weights}_{scope}.json")
        viable_solution = curve["is_eligible_solution"]
        f1s = curve["f1_values"]
        f2s = curve["f2_values"]
        eligible_f1s = [f1 for index, f1 in enumerate(f1s) if viable_solution[index]]
        eligible_f2s = [f2 for index, f2 in enumerate(f2s) if viable_solution[index]]
        non_viable_f1.extend([f1 for index, f1 in enumerate(f1s) if not viable_solution[index]])
        non_viable_f2.extend([f2 for index, f2 in enumerate(f2s) if not viable_solution[index]])
        all_points.extend([Coordinate(x=x, y=eligible_f2s[index]) for index, x in enumerate(eligible_f1s)])
        plt.plot(eligible_f1s, eligible_f2s, '.', label=f'Execution {curve_number}')
    pareto_greats = CoordinatesCalculator.find_pareto_frontier(points=all_points)
    plt.plot([p.x for p in pareto_greats], [p.y for p in pareto_greats], 'k^', label=f'Pareto greats')
    plt.plot(curve["f1_values"][:2], curve["f2_values"][:2], 'bs', label=f'Mono optimization')
    plt.plot(non_viable_f1, non_viable_f2, 'rx', label=f'Non viable solutions')
    plt.legend()
    plt.savefig(fname=f'{scope}_results/{scope}_{n}_{n_curves}_{n_weights}_pareto_great.png')
    plt.close()


if __name__ == '__main__':
    plot_result(scope="pondered_sum")
    # is_eligible_solution = []
    # problem_f1 = ProblemDefinitionF1.from_csv()
    # rvns_f1 = Rvns(problem=problem_f1, max_solutions_evaluations=5 * N_EVALUATIONS, n=1)
    # rvns_f1.run()
    # problem_f2 = ProblemDefinitionF2.from_csv()
    # rvns_f2 = Rvns(problem=problem_f2, max_solutions_evaluations=5 * N_EVALUATIONS, n=1)
    # rvns_f2.run()
    # f1_min_solution = rvns_f1.best_solution
    # f2_min_solution = rvns_f2.best_solution
    # for iteration in range(N_CURVES):
    #     is_eligible_solution = [True, True]
    #     f1_values = [47.0, 100]
    #     f2_values = [6351.853168212587, 19650.015388388183]
    #     for _ in range(N_WEIGHTS):
    #         w1 = numpy.random.uniform(0, 1)
    #         problem = PonderedSumProblem.from_csv(w1=w1, w2=1 - w1)
    #         rvns = Rvns(problem=problem, max_solutions_evaluations=N_EVALUATIONS, n=1, kmax=5)
    #         rvns.run()
    #         f1_values.append(len(rvns.best_solution.active_points))
    #         f2_values.append(rvns.best_solution.total_distance)
    #         is_eligible_solution.append(rvns.best_solution.penal == 0)
    #     save_file(file_name=f"execution_{iteration}_{N_EVALUATIONS}_{N_CURVES}_{N_WEIGHTS}_pondered_sum.json",
    #               f1_values=f1_values, f2_values=f2_values, is_eligible_solution=is_eligible_solution)
