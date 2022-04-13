from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem import Srinivas
from jmetal.problem.multiobjective.cdtlz import C1_DTLZ1, C1_DTLZ3, C2_DTLZ2, C3_DTLZ1, C3_DTLZ4
from jmetal.problem.multiobjective.lircmop import *
from jmetal.util.observer import VisualizerObserver, ProgressBarObserver
from jmetal.util.solutions.comparator import DominanceComparator
from jmetal.util.solutions import read_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = Srinivas()
    problem.reference_front = read_solutions(filename='resources/reference_front/Srinivas.pf')

    problem = C3_DTLZ4()
    problem.reference_front = read_solutions(filename='../../../resources/reference_front/C3_DTLZ4.3D.pf')

    problem = LIRCMOP13()
    problem.reference_front = read_solutions(filename='../../../resources/reference_front/LIRCMOP13.pf')

    max_evaluations = 25000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max=max_evaluations),
        dominance_comparator=DominanceComparator()
    )
    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(
        observer=VisualizerObserver(reference_front=problem.reference_front))

    algorithm.run()
    front = algorithm.get_result()

    # Save results to file
    print_function_values_to_file(front, 'test/FUN.' + algorithm.label)
    print_variables_to_file(front, 'test/VAR.'+ algorithm.label)

    print(f'Algorithm: ${algorithm.get_name()}')
    print(f'Problem: ${problem.get_name()}')
    print(f'Computing time: ${algorithm.total_computing_time}')
