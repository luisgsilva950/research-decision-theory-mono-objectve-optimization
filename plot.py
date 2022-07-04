from matplotlib import pyplot as plt

from models import Coordinate, CoordinatesCalculator, AccessPoint
from multi_objective_problem_epsilon import EpsilonRestrictProblem
from utils import load_file, save_file

if __name__ == '__main__':
    curve = load_file('epsilon_restrict_results/execution_0_300_5_epsilon_restrict.json')
    problem = EpsilonRestrictProblem.from_csv(max_active_points=100)
    problem.solution = curve["solution"][9]
    problem.update_active_points()
    problem.plot_solution()
    # non_viable_f1 = []
    # non_viable_f2 = []
    # all_points = []
    # plt.xlabel("F1")
    # plt.ylabel("F2")
    # viable_solution = curve["is_eligible_solution"]
    # f1s = curve["f1_values"]
    # f2s = curve["f2_values"]
    # distances_per_execution = curve["distance"]
    # capacities_per_execution = curve["consumed_capacity"]
    # eligible_f1s = [f1 for index, f1 in enumerate(f1s) if viable_solution[index]]
    # eligible_f2s = [f2 for index, f2 in enumerate(f2s) if viable_solution[index]]
    # capacities = [capacities_per_execution[index] for index, f2 in enumerate(eligible_f1s)]
    # distances = [distances_per_execution[index] for index, f2 in enumerate(eligible_f1s)]
    # non_viable_f1.extend([f1 for index, f1 in enumerate(f1s) if not viable_solution[index]])
    # non_viable_f2.extend([f2 for index, f2 in enumerate(f2s) if not viable_solution[index]])
    # all_points.extend([AccessPoint(x=x, y=eligible_f2s[index], index=index) for index, x in enumerate(eligible_f1s)])
    # plt.plot(eligible_f1s, eligible_f2s, '.', label=f'Execution {0}')
    # pareto_greats = CoordinatesCalculator.get_non_dominated_coordinates(points=all_points)
    # plt.plot([p.x for p in pareto_greats], [p.y for p in pareto_greats], 'k^', label=f'Pareto greats')
    # plt.plot(curve["f1_values"][:2], curve["f2_values"][:2], 'bs', label=f'Mono optimization')
    # plt.plot(non_viable_f1, non_viable_f2, 'rx', label=f'Non viable solutions')
    # save_file('teste_pareto.json',
    #           paretos=[dict(coordinates=p.to_dict(), distance=distances[p.index], capacity=capacities[p.index]) for p in
    #                    pareto_greats])
    # plt.legend()
    # plt.show()
    # plt.savefig(fname=f'teste.png')
    # plt.close()
