from problem_f1 import ProblemDefinitionF1
from problem_f2 import ProblemDefinitionF2
from rvns import Rvns

if __name__ == '__main__':
    problem = ProblemDefinitionF2.from_csv()
    rvns_f2 = Rvns(problem=problem, max_solutions_evaluations=100)
    rvns_f2.run()
