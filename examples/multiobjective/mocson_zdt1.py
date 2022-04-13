import pprint

from jmetal.algorithm.multiobjective.mocso_nsga2 import MOCSON
from jmetal.lab.visualization import InteractivePlot, Plot
from jmetal.operator import UniformMutation, SBXCrossover
from jmetal.operator.mutation import NonUniformMutation, PolynomialMutation
from jmetal.problem import *
from jmetal.util.archive import CrowdingDistanceArchive, GDominanceComparator, DominanceComparator
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solutions import print_function_values_to_file, print_variables_to_file
from jmetal.util.solutions import read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    # problem = ZDT1()
    # reference_point = [[1.0, 1.0]]
    # problem.reference_front = read_solutions(filename='../../resources/reference_front/ZDT1.pf')
    # problem = ZDT4()
    # reference_point = [[1.0, 1.0]]
    # problem.reference_front = read_solutions(filename='../../resources/reference_front/ZDT4.pf')
    # problem = Schaffer()
    # problem.reference_front = read_solutions(filename='resources/reference_front/Schaffer.pf')
    # problem = DTLZ7()
    # reference_point = [[1.0, 1.0, 1.0]]
    # problem.reference_front = read_solutions(filename='../../resources/reference_front/DTLZ7.3D.pf')
    # problem = DTLZ2()
    # problem.reference_front = read_solutions(filename='../../resources/reference_front/DTLZ2.3D.pf')
    #
    problem = DTLZ3()
    reference_point = [[1.0, 1.0, 1.0]]
    problem.reference_front = read_solutions(filename='../../resources/reference_front/DTLZ3.3D.pf')
    mutation_probability = 1.0 / problem.number_of_variables
    # max_evaluations = 50000 #最大迭代次数
    # swarm_size = 100
    max_evaluations = 25000  # 最大迭代次数
    swarm_size = 100

    algorithm = MOCSON(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        dominance_comparator=DominanceComparator(),
        epsilon=0.0075,
        uniform_mutation=UniformMutation(probability=mutation_probability, perturbation=0.5),
        non_uniform_mutation=NonUniformMutation(mutation_probability, perturbation=0.5,
                                                max_iterations=int(max_evaluations / swarm_size)),
        leaders=CrowdingDistanceArchive(100),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(
        observer=VisualizerObserver(reference_front=problem.reference_front, reference_point=reference_point))

    algorithm.run()
    front = algorithm.get_result()
    algorithm.show_size_archieve()
    pp = pprint.PrettyPrinter(indent=4)
    solutionCrowd = []
    num = 0
    # for solution in front[1:]:
    #     # pareto前沿粒子的拥挤距离数组
    #     if (solution.attributes.get("crowding_distance")>0.0075):
    #         num += 1
    #     solutionCrowd.append(solution.attributes.get("crowding_distance"))
    # pp.pprint(solutionCrowd)
    # pp.pprint(num)
    # pp.pprint(num/len(front))
    # # Plot front
    # plot_front = Plot(plot_title='Pareto front approximation', reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    # plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())
    #
    # # Plot interactive front
    # plot_front = InteractivePlot(plot_title='Pareto front approximation', reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    # plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())

    # Save results to file
    print_function_values_to_file(front, "./result/" + 'FUN.' + algorithm.label)
    print_variables_to_file(front, "./result/" + 'VAR.' + algorithm.label)

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
