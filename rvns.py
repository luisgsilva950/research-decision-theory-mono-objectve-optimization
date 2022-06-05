import numpy as np
from matplotlib import pyplot as plt

from models import ProblemDefinition


class Rvns:
    def __init__(self, problem: ProblemDefinition, kmax: int = 3, max_solutions_evaluations: int = 1000):
        self.problem = problem
        self.kmax = kmax
        self.max_solutions_evaluations = max_solutions_evaluations
        self.penal_fitness_historic = []
        self.penal_historic = []

    def run(self):
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        self.problem = self.problem.get_initial_solution()
        num_evaluated_solutions = 0
        print(f'Initial Fitness: {self.problem.penal_fitness}\n')
        while num_evaluated_solutions <= self.max_solutions_evaluations:
            self.problem.k = 1
            while self.problem.k <= self.kmax:
                new_solution = self.problem.shake()
                new_solution = new_solution.objective_function()
                num_evaluated_solutions += 1
                will_change = new_solution.penal_fitness < self.problem.penal_fitness
                self.problem = self.problem.neighborhood_change(y=new_solution)
                self.penal_fitness_historic.append(self.problem.penal_fitness)
                self.penal_historic.append(self.problem.penal)
                print(f'\033[3;{"92" if will_change else "91"}m'
                      f'Penal fitness {num_evaluated_solutions}, k: {self.problem.k - 1}: {self.problem.penal_fitness}, '
                      f'penal: {self.problem.penal}\n')
        s = len(self.penal_fitness_historic)
        ax1.plot(np.linspace(0, s - 1, s), self.penal_fitness_historic, 'k-')
        ax2.plot(np.linspace(0, s - 1, s), self.penal_historic, 'b:')
        fig.suptitle("Evolution's quality of candidate solution")
        ax1.set_ylabel('Penal fitness(x)')
        ax2.set_ylabel('Penal(x)')
        ax2.set_xlabel('Number of evaluations')
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        plt.show()
        self.problem.plot_solution()
