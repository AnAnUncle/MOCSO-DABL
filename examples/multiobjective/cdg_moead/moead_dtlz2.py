from jmetal.algorithm.multiobjective.cdg import CDGMOEAD
from jmetal.core.quality_indicator import HyperVolume, InvertedGenerationalDistance
from jmetal.operator import PolynomialMutation, DifferentialEvolutionCrossover
from jmetal.problem import *
from jmetal.problem.multiobjective.unconstrained import SchafferInteger
from jmetal.util.aggregative_function import Tschebycheff
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solutions import read_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = DTLZ2()
    problem.reference_front = read_solutions(filename='../../../resources/reference_front/DTLZ2.3D.pf')
    problem = ZDT4()
    problem.reference_front = read_solutions(filename='../../../resources/reference_front/ZDT4.pf')
    # problem = SchafferInteger()
    # problem.reference_front = read_solutions(filename='../../../resources/reference_front/Schaffer.pf')

    max_evaluations = 50000

    algorithm = CDGMOEAD(
        problem=problem,
        population_size=100,
        crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5, K=0.5),
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        K=5,
        T=1,
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )
    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(
        observer=VisualizerObserver(reference_front=problem.reference_front))
    algorithm.run()
    front = algorithm.get_result()

    # hypervolume = HyperVolume([1.0, 1.0, 1.0])
    # print("Hypervolume: " + str(hypervolume.compute([front[i].objectives for i in range(len(front))])))

    # Save results to file
    # print_function_values_to_file(front, 'FUN.' + algorithm.label)
    # print_variables_to_file(front, 'VAR.'+ algorithm.label)

    print(f'Algorithm: ${algorithm.get_name()}')
    print(f'Problem: ${problem.get_name()}')
    print(f'Computing time: ${algorithm.total_computing_time}')
