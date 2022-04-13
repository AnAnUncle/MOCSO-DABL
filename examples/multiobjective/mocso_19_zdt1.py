import pprint

from jmetal.algorithm.multiobjective.mocso_19 import MOCSO19
from jmetal.lab.visualization import InteractivePlot, Plot
from jmetal.operator import UniformMutation
from jmetal.operator.mutation import NonUniformMutation, PolynomialMutation
from jmetal.problem import *
from jmetal.util.archive_mocso_19 import CrowdingDistanceArchive, CrowdingDistanceArchive19
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solutions import print_function_values_to_file, print_variables_to_file
from jmetal.util.solutions import read_solutions
from jmetal.util.solutions.helper import read_port_reference
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    # reference_point = [[1.0, 1.0]]
    problem = ZDT1()
    problem.reference_front = read_solutions(filename='../../resources/reference_front/ZDT1.pf')
    # problem = ZDT2()
    # problem.reference_front = read_solutions(filename='../../resources/reference_front/ZDT2.pf')
    # problem = ZDT3()
    # problem.reference_front = read_solutions(filename='../../resources/reference_front/ZDT3.pf')
    # problem = ZDT4()
    # problem.reference_front = read_solutions(filename='../../resources/reference_front/ZDT4.pf')
    # problem = ZDT6()
    # problem.reference_front = read_solutions(filename='../../resources/reference_front/ZDT6.pf')
    # problem = Schaffer()
    # problem.reference_front = read_solutions(filename='../../resources/reference_front/Schaffer.pf')
    # problem = Fonseca()
    # problem.reference_front = read_solutions(filename='../../resources/reference_front/Fonseca.pf')
    # problem = Kursawe()
    # problem.reference_front = read_solutions(filename='../../resources/reference_front/Kursawe.pf')

    # reference_point = [[1.0, 1.0, 1.0]]
    # problem = DTLZ1()
    # problem.reference_front = read_solutions(filename='../../resources/reference_front/DTLZ1.3D.pf')
    # problem = DTLZ2()
    # problem.reference_front = read_solutions(filename='../../resources/reference_front/DTLZ2.3D.pf')
    # problem = DTLZ3()
    # problem.reference_front = read_solutions(filename='../../resources/reference_front/DTLZ3.3D.pf')
    # problem = DTLZ4()
    # problem.reference_front = read_solutions(filename='../../resources/reference_front/DTLZ4.3D.pf')
    # problem = DTLZ5()
    # problem.reference_front = read_solutions(filename='../../resources/reference_front/DTLZ5.3D.pf')
    # problem = DTLZ6()
    # # problem.reference_front = read_solutions(filename='../../resources/reference_front/DTLZ6.3D.pf')
    # problem = DTLZ7()
    # problem.reference_front = read_solutions(filename='../../resources/reference_front/DTLZ7.3D.pf')
    # problem = Viennet2()
    # problem.reference_front = read_solutions(filename='../../resources/reference_front/Viennet2.3D.pf')

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

    mutation_probability = 1.0 / problem.number_of_variables
    # max_evaluations = 50000 #最大迭代次数
    # swarm_size = 100
    max_evaluations = 20000  # 最大迭代次数
    swarm_size = 100

    algorithm = MOCSO19(
        problem=problem,
        swarm_size=swarm_size,
        archive=CrowdingDistanceArchive19(100),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(
        observer=VisualizerObserver(reference_front=problem.reference_front))
    # algorithm.observable.register(
    #     observer=VisualizerObserver(reference_front=problem.reference_front, reference_point=reference_point))
    # algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    # algorithm.observable.register(observer=VisualizerObserver())

    algorithm.run()
    front = algorithm.get_result()
    algorithm.show_size_archieve()
    pp = pprint.PrettyPrinter(indent=4)
    solutionCrowd = []
    num = 0

    # Save results to file
    print_function_values_to_file(front, "./result/" + 'FUN.' + algorithm.label)
    print_variables_to_file(front, "./result/" + 'VAR.' + algorithm.label)

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
