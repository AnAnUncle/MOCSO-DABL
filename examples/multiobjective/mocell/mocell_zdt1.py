from jmetal.algorithm.multiobjective.mocell import MOCell
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem import ZDT1
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.neighborhood import C9
from jmetal.util.solutions import read_solutions

from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = ZDT1()
    problem.reference_front = read_solutions(filename='../../../resources/reference_front/ZDT4.pf')

    max_evaluations = 25000
    algorithm = MOCell(
        problem=problem,
        population_size=100,
        neighborhood=C9(10, 10),
        archive=CrowdingDistanceArchive(100),
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )
    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(
        observer=VisualizerObserver(reference_front=problem.reference_front))
    algorithm.run()
    front = algorithm.get_result()

    # Save results to file
    # print_function_values_to_file(front, 'FUN.' + algorithm.label)
    # print_variables_to_file(front, 'VAR.'+ algorithm.label)

    print(f'Algorithm: ${algorithm.get_name()}')
    print(f'Problem: ${problem.get_name()}')
    print(f'Computing time: ${algorithm.total_computing_time}')
