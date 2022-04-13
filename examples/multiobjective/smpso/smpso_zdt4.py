from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.operator import PolynomialMutation
from jmetal.problem import *
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solutions import print_function_values_to_file, print_variables_to_file
from jmetal.util.solutions import read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    # problem = ZDT4()
    # problem.reference_front = read_solutions(filename='../../../resources/reference_front/ZDT4.pf')
    # problem = DTLZ1()
    # problem.reference_front = read_solutions(filename='../../resources/reference_front/DTLZ1.3D.pf')
    # mutation_probability = 1.0 / problem.number_of_variables
    # problem = DTLZ1()
    # problem.reference_front = read_solutions(filename='../../../resources/reference_front/DTLZ1.3D.pf')
    problem = Psp31()
    mutation_probability = 1.0 / problem.number_of_variables

    max_evaluations = 25000
    algorithm = SMPSO(
        problem=problem,
        swarm_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        leaders=CrowdingDistanceArchive(100),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )
    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    # algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front))
    algorithm.observable.register(observer=VisualizerObserver())
    algorithm.run()
    front = algorithm.get_result()

    # Save results to file
    # print_function_values_to_file(front, "./result/" + 'FUN.' + algorithm.label)
    # print_variables_to_file(front, "./result/" + 'VAR.' + algorithm.label)

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
