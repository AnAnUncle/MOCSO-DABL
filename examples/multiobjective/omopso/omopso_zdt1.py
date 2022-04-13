from jmetal.algorithm.multiobjective.omopso import OMOPSO
from jmetal.operator import UniformMutation
from jmetal.operator.mutation import NonUniformMutation
from jmetal.problem import *
from jmetal.util.archive_omopso import CrowdingDistanceArchiveOmopso
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solutions import print_function_values_to_file, print_variables_to_file
from jmetal.util.solutions import read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = DTLZ7()
    problem.reference_front = read_solutions(filename='../../../resources/reference_front/DTLZ7.3D.pf')
    # problem = ZDT4()
    # problem.reference_front = read_solutions(filename='../../../resources/reference_front/ZDT4.pf')
    # problem = ZDT1()
    # problem.reference_front = read_solutions(filename='../../../resources/reference_front/ZDT1.pf')
    # problem = DTLZ1()
    # problem.reference_front = read_solutions(filename='../../../resources/reference_front/DTLZ1.3D.pf')

    mutation_probability = 1.0 / problem.number_of_variables
    # max_evaluations = 25000
    # swarm_size = 100
    max_evaluations = 15000  # 最大迭代次数
    swarm_size = 100

    algorithm = OMOPSO(
        problem=problem,
        swarm_size=swarm_size,
        epsilon=0.0075,
        uniform_mutation=UniformMutation(probability=mutation_probability, perturbation=0.5),
        non_uniform_mutation=NonUniformMutation(mutation_probability, perturbation=0.5,
                                                max_iterations=int(max_evaluations / swarm_size)),
        leaders=CrowdingDistanceArchiveOmopso(100),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )
    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front))

    algorithm.run()
    front = algorithm.get_result()

    # # Save results to file
    # print_function_values_to_file(front, "./result/" + 'FUN.' + algorithm.label)
    # print_variables_to_file(front, "./result/" + 'VAR.' + algorithm.label)

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
