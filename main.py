from problem_f1 import ProblemDefinitionF1
from rvns import Rvns

if __name__ == '__main__':
    problem = ProblemDefinitionF1.from_csv()
    rvns_f1 = Rvns(problem=problem, max_solutions_evaluations=150)
    rvns_f1.run()
