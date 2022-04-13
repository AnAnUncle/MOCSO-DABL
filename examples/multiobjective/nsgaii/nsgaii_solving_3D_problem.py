from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem import *
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solutions.comparator import DominanceComparator
from jmetal.util.solutions import print_function_values_to_file, print_variables_to_file, read_solutions
from jmetal.util.solutions.helper import read_port_reference
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    reference_point = [[1.0, 1.0]]
    problem = ZDT1()
    problem.reference_front = read_solutions(filename='../../../resources/reference_front/ZDT1.pf')
    # problem = ZDT2()
    # problem.reference_front = read_solutions(filename='../../../resources/reference_front/ZDT2.pf')
    # problem = ZDT3()
    # problem.reference_front = read_solutions(filename='../../../resources/reference_front/ZDT3.pf')
    # problem = ZDT4()
    # problem.reference_front = read_solutions(filename='../../../resources/reference_front/ZDT4.pf')
    # problem = ZDT6()
    # problem.reference_front = read_solutions(filename='../../../resources/reference_front/ZDT6.pf')
    # problem = Schaffer()
    # problem.reference_front = read_solutions(filename='../../../resources/reference_front/Schaffer.pf')
    # problem = Fonseca()
    # problem.reference_front = read_solutions(filename='../../../resources/reference_front/Fonseca.pf')

    # reference_point = [[4.0, 1.0, 6.0]]
    # problem = DTLZ1()
    # problem.reference_front = read_solutions(filename='../../../resources/reference_front/DTLZ1.3D.pf')
    # problem = DTLZ2()
    # problem.reference_front = read_solutions(filename='../../../resources/reference_front/DTLZ2.3D.pf')
    # problem = DTLZ3()
    # problem.reference_front = read_solutions(filename='../../../resources/reference_front/DTLZ3.3D.pf')
    # problem = DTLZ4()
    # problem.reference_front = read_solutions(filename='../../../resources/reference_front/DTLZ4.3D.pf')
    # problem = DTLZ5()
    # problem.reference_front = read_solutions(filename='../../../resources/reference_front/DTLZ5.3D.pf')
    # problem = DTLZ6()
    # problem.reference_front = read_solutions(filename='../../../resources/reference_front/DTLZ6.3D.pf')
    # problem = DTLZ7()
    # problem.reference_front = read_solutions(filename='../../../resources/reference_front/DTLZ7.3D.pf')

    # problem = Psp31()
    # problem.reference_front = read_port_reference(filename='../../resources/reference_front/Psp31.txt')
    # problem = Psp85()
    # problem.reference_front = read_port_reference(filename='../../resources/reference_front/Psp85.txt')
    # problem = Psp89()
    # problem.reference_front = read_port_reference(filename='../../resources/reference_front/Psp89.txt')
    # problem = Psp98()
    # problem.reference_front = read_port_reference(filename='../../resources/reference_front/Psp98.txt')
    # problem = Psp225()
    # problem.reference_front = read_port_reference(filename='../../resources/reference_front/Psp225.txt')

    max_evaluations = 20000
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
    algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front))

    algorithm.run()
    front = algorithm.get_result()

    # # Save results to file
    # print_function_values_to_file(front, "./result/" + 'FUN.' + algorithm.label)
    # print_variables_to_file(front, "./result/" + 'VAR.' + algorithm.label)

    print(f'Algorithm: ${algorithm.get_name()}')
    print(f'Problem: ${problem.get_name()}')
    print(f'Computing time: ${algorithm.total_computing_time}')
