from jmetal.algorithm.multiobjective.ibea import IBEA
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem import *
from jmetal.problem.multiobjective.lircmop import LIRCMOP13
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solutions import read_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = ZDT4()
    problem.reference_front = read_solutions(filename='../../../resources/reference_front/ZDT4.pf')
    problem = DTLZ7()
    problem.reference_front = read_solutions(filename='../../../resources/reference_front/DTLZ7.3D.pf')
    problem = LIRCMOP13()
    problem.reference_front = read_solutions(filename='../../../resources/reference_front/LIRCMOP13.pf')
    reference_point = [0.2, 0.5]

    max_evaluations = 15000

    algorithm = IBEA(
        problem=problem,
        kappa=1.,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(
        observer=VisualizerObserver(reference_front=problem.reference_front, reference_point=reference_point))

    algorithm.run()
    front = algorithm.get_result()

    # Save results to file
    # print_function_values_to_file(front, 'FUN.' + algorithm.label)
    # print_variables_to_file(front, 'VAR.'+ algorithm.label)

    print(f'Algorithm: ${algorithm.get_name()}')
    print(f'Problem: ${problem.get_name()}')
    print(f'Computing time: ${algorithm.total_computing_time}')
