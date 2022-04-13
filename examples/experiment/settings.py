from jmetal.algorithm.multiobjective.gde3 import GDE3
from jmetal.algorithm.multiobjective.ibea import IBEA
from jmetal.algorithm.multiobjective.mocso import MOCSO
from jmetal.algorithm.multiobjective.mocso_19 import MOCSO19
# from jmetal.algorithm.multiobjective.moead import MOEAD
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.nsgaiii import UniformReferenceDirectionFactory, NSGAIII
from jmetal.algorithm.multiobjective.omopso import OMOPSO
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.lab.experiment import Job
from jmetal.operator import PolynomialMutation, SBXCrossover, UniformMutation, DifferentialEvolutionCrossover
from jmetal.operator.mutation import NonUniformMutation
from jmetal.util.aggregative_function import Tschebycheff
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.archive_mocso_19 import CrowdingDistanceArchive19
from jmetal.util.archive_omopso import CrowdingDistanceArchiveOmopso
from jmetal.util.termination_criterion import StoppingByEvaluations
import datetime

experiment_tag = "all_test_constrain3d2.0"
# max_evaluations = 15000
max_evaluations = 20000
# max_evaluations = 25000
# max_evaluations = 30000

tagDate = datetime.date.today()
tagHour = datetime.datetime.now().hour
tagMinute = datetime.datetime.now().minute
tagTime = 'analysis_data'


def configure_experiment_cso(problems: dict, n_run: int, max_evaluations: int):
    jobs = []
    for run in range(n_run):
        for problem_tag, problem in problems.items():
            jobs.append(
                Job(
                    algorithm=MOCSO(
                        problem=problem,
                        swarm_size=100,
                        uniform_mutation=UniformMutation(probability=1.0 / problem.number_of_variables,
                                                         perturbation=0.5),
                        non_uniform_mutation=NonUniformMutation(1.0 / problem.number_of_variables, perturbation=0.5,
                                                                max_iterations=int(max_evaluations / 100)),
                        # mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
                        leaders=CrowdingDistanceArchive(100),
                        termination_criterion=StoppingByEvaluations(max=max_evaluations)
                    ),
                    algorithm_tag='DAMOCSO',
                    problem_tag=problem_tag,
                    run=run,
                )
            )

    return jobs


def configure_experiment_cso19(problems: dict, n_run: int, max_evaluations: int):
    jobs = []
    for run in range(n_run):
        for problem_tag, problem in problems.items():
            jobs.append(
                Job(
                    algorithm=MOCSO19(
                        problem=problem,
                        swarm_size=100,
                        archive=CrowdingDistanceArchive19(100),
                        termination_criterion=StoppingByEvaluations(max=max_evaluations)
                    ),
                    algorithm_tag='DAMOCSO19',
                    problem_tag=problem_tag,
                    run=run,
                )
            )

    return jobs


def configure_experiment_psp(problems: dict, n_run: int, max_evaluations: int):
    jobs = []
    for run in range(n_run):
        for problem_tag, problem in problems.items():
            jobs.append(
                Job(
                    algorithm=MOCSO(
                        problem=problem,
                        swarm_size=100,
                        uniform_mutation=UniformMutation(probability=1.0 / problem.number_of_variables,
                                                         perturbation=0.5),
                        non_uniform_mutation=NonUniformMutation(1.0 / problem.number_of_variables, perturbation=0.5,
                                                                max_iterations=int(max_evaluations / 100)),
                        # mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
                        leaders=CrowdingDistanceArchive(100),
                        termination_criterion=StoppingByEvaluations(max=max_evaluations)
                    ),
                    algorithm_tag='DAMOCSO',
                    problem_tag=problem_tag,
                    run=run,
                )
            )
            jobs.append(
                Job(
                    algorithm=NSGAII(
                        problem=problem,
                        population_size=100,
                        offspring_population_size=100,
                        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
                                                    distribution_index=20),
                        crossover=SBXCrossover(probability=1.0, distribution_index=20),
                        termination_criterion=StoppingByEvaluations(max=max_evaluations)
                    ),
                    algorithm_tag='NSGAII',
                    problem_tag=problem_tag,
                    run=run,
                )
            )

    return jobs


def configure_experiment_2d(problems: dict, n_run: int, max_evaluations: int):
    jobs = []
    for run in range(n_run):
        for problem_tag, problem in problems.items():
            # jobs.append(
            #     Job(
            #         algorithm=SMPSO(
            #             problem=problem,
            #             swarm_size=100,
            #             mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
            #                                         distribution_index=20),
            #             leaders=CrowdingDistanceArchive(100),
            #             termination_criterion=StoppingByEvaluations(max=max_evaluations)
            #         ),
            #         algorithm_tag='SMPSO',
            #         problem_tag=problem_tag,
            #         run=run,
            #     )
            # )
            # jobs.append(
            #     Job(
            #         algorithm=SPEA2(
            #             problem=problem,
            #             population_size=100,
            #             offspring_population_size=100,
            #             mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
            #                                         distribution_index=20),
            #             crossover=SBXCrossover(probability=1.0, distribution_index=20),
            #             termination_criterion=StoppingByEvaluations(max=max_evaluations)
            #         ),
            #         algorithm_tag='SPEA2',
            #         problem_tag=problem_tag,
            #         run=run,
            #     )
            # )
            # jobs.append(
            #     Job(
            #         MOEAD(
            #             problem=problem,
            #             population_size=100,
            #             crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5, K=0.5),
            #             mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
            #                                         distribution_index=20),
            #             aggregative_function=Tschebycheff(dimension=problem.number_of_objectives),
            #             neighbor_size=20,
            #             neighbourhood_selection_probability=0.9,
            #             max_number_of_replaced_solutions=2,
            #             weight_files_path='../../resources/MOEAD_weights',
            #             termination_criterion=StoppingByEvaluations(max=max_evaluations),
            #
            #         ),
            #         algorithm_tag='MOEAD',
            #         problem_tag=problem_tag,
            #         run=run,
            #     )
            # )
            jobs.append(
                Job(
                    algorithm=MOCSO(
                        problem=problem,
                        swarm_size=100,
                        uniform_mutation=UniformMutation(probability=1.0 / problem.number_of_variables,
                                                         perturbation=0.5),
                        non_uniform_mutation=NonUniformMutation(1.0 / problem.number_of_variables, perturbation=0.5,
                                                                max_iterations=int(max_evaluations / 100)),
                        # mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
                        leaders=CrowdingDistanceArchive(100),
                        termination_criterion=StoppingByEvaluations(max=max_evaluations)
                    ),
                    algorithm_tag='DAMOCSO',
                    problem_tag=problem_tag,
                    run=run,
                )
            )
            jobs.append(
                Job(
                    algorithm=NSGAII(
                        problem=problem,
                        population_size=100,
                        offspring_population_size=100,
                        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
                                                    distribution_index=20),
                        crossover=SBXCrossover(probability=1.0, distribution_index=20),
                        termination_criterion=StoppingByEvaluations(max=max_evaluations)
                    ),
                    algorithm_tag='NSGAII',
                    problem_tag=problem_tag,
                    run=run,
                )
            )
            jobs.append(
                Job(
                    algorithm=NSGAIII(
                        problem=problem,
                        population_size=100,
                        reference_directions=UniformReferenceDirectionFactory(2, n_points=100),
                        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
                                                    distribution_index=20),
                        crossover=SBXCrossover(probability=1.0, distribution_index=30),
                        termination_criterion=StoppingByEvaluations(max=max_evaluations)
                    ),
                    algorithm_tag='NSGAIII',
                    problem_tag=problem_tag,
                    run=run,
                )
            )
            jobs.append(
                Job(
                    IBEA(
                        problem=problem,
                        kappa=1.,
                        population_size=100,
                        offspring_population_size=100,
                        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
                                                    distribution_index=20),
                        crossover=SBXCrossover(probability=1.0, distribution_index=20),
                        termination_criterion=StoppingByEvaluations(max=max_evaluations)
                    ),
                    algorithm_tag='IBEA',
                    problem_tag=problem_tag,
                    run=run,
                )
            )
            jobs.append(
                Job(
                    OMOPSO(
                        problem=problem,
                        swarm_size=100,
                        epsilon=0.0075,
                        uniform_mutation=UniformMutation(probability=1.0 / problem.number_of_variables,
                                                         perturbation=0.5),
                        non_uniform_mutation=NonUniformMutation(probability=1.0 / problem.number_of_variables,
                                                                perturbation=0.5,
                                                                max_iterations=int(max_evaluations / 100)),
                        leaders=CrowdingDistanceArchiveOmopso(100),
                        termination_criterion=StoppingByEvaluations(max=max_evaluations)
                    ),
                    algorithm_tag='OMOPSO',
                    problem_tag=problem_tag,
                    run=run,
                )
            )
            jobs.append(
                Job(
                    algorithm=GDE3(
                        problem=problem,
                        population_size=100,
                        cr=0.5,
                        f=0.5,
                        termination_criterion=StoppingByEvaluations(max=max_evaluations)
                    ),
                    algorithm_tag='GDE3',
                    problem_tag=problem_tag,
                    run=run,
                )
            )

    return jobs


def configure_experiment_3d(problems: dict, n_run: int, max_evaluations: int):
    jobs = []
    for run in range(n_run):
        for problem_tag, problem in problems.items():
            # jobs.append(
            #     Job(
            #         algorithm=SMPSO(
            #             problem=problem,
            #             swarm_size=100,
            #             mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
            #                                         distribution_index=20),
            #             leaders=CrowdingDistanceArchive(100),
            #             termination_criterion=StoppingByEvaluations(max=max_evaluations)
            #         ),
            #         algorithm_tag='SMPSO',
            #         problem_tag=problem_tag,
            #         run=run,
            #     )
            # )
            # jobs.append(
            #     Job(
            #         algorithm=SPEA2(
            #             problem=problem,
            #             population_size=100,
            #             offspring_population_size=100,
            #             mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
            #                                         distribution_index=20),
            #             crossover=SBXCrossover(probability=1.0, distribution_index=20),
            #             termination_criterion=StoppingByEvaluations(max=max_evaluations)
            #         ),
            #         algorithm_tag='SPEA2',
            #         problem_tag=problem_tag,
            #         run=run,
            #     )
            # )
            # jobs.append(
            #     Job(
            #         MOEAD(
            #             problem=problem,
            #             population_size=100,
            #             crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5, K=0.5),
            #             mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
            #                                         distribution_index=20),
            #             aggregative_function=Tschebycheff(dimension=problem.number_of_objectives),
            #             neighbor_size=20,
            #             neighbourhood_selection_probability=0.9,
            #             max_number_of_replaced_solutions=2,
            #             weight_files_path='../../resources/MOEAD_weights',
            #             termination_criterion=StoppingByEvaluations(max=max_evaluations),
            #
            #         ),
            #         algorithm_tag='MOEAD',
            #         problem_tag=problem_tag,
            #         run=run,
            #     )
            # )
            jobs.append(
                Job(
                    algorithm=MOCSO(
                        problem=problem,
                        swarm_size=100,
                        uniform_mutation=UniformMutation(probability=1.0 / problem.number_of_variables,
                                                         perturbation=0.5),
                        non_uniform_mutation=NonUniformMutation(1.0 / problem.number_of_variables, perturbation=0.5,
                                                                max_iterations=int(max_evaluations / 100)),
                        leaders=CrowdingDistanceArchive(100),
                        termination_criterion=StoppingByEvaluations(max=max_evaluations)
                    ),
                    algorithm_tag='DAMOCSO',
                    problem_tag=problem_tag,
                    run=run,
                )
            )
            jobs.append(
                Job(
                    algorithm=NSGAII(
                        problem=problem,
                        population_size=100,
                        offspring_population_size=100,
                        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
                                                    distribution_index=20),
                        crossover=SBXCrossover(probability=1.0, distribution_index=20),
                        termination_criterion=StoppingByEvaluations(max=max_evaluations)
                    ),
                    algorithm_tag='NSGAII',
                    problem_tag=problem_tag,
                    run=run,
                )
            )
            jobs.append(
                Job(
                    algorithm=NSGAIII(
                        problem=problem,
                        population_size=100,
                        reference_directions=UniformReferenceDirectionFactory(3, n_points=100),
                        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
                                                    distribution_index=20),
                        crossover=SBXCrossover(probability=1.0, distribution_index=30),
                        termination_criterion=StoppingByEvaluations(max=max_evaluations)
                    ),
                    algorithm_tag='NSGAIII',
                    problem_tag=problem_tag,
                    run=run,
                )
            )
            # jobs.append(
            #     Job(
            #         IBEA(
            #             problem=problem,
            #             kappa=1.,
            #             population_size=100,
            #             offspring_population_size=100,
            #             mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
            #                                         distribution_index=20),
            #             crossover=SBXCrossover(probability=1.0, distribution_index=20),
            #             termination_criterion=StoppingByEvaluations(max=max_evaluations)
            #         ),
            #         algorithm_tag='IBEA',
            #         problem_tag=problem_tag,
            #         run=run,
            #     )
            # )
            jobs.append(
                Job(
                    OMOPSO(
                        problem=problem,
                        swarm_size=100,
                        epsilon=0.0075,
                        uniform_mutation=UniformMutation(probability=1.0 / problem.number_of_variables,
                                                         perturbation=0.5),
                        non_uniform_mutation=NonUniformMutation(probability=1.0 / problem.number_of_variables,
                                                                perturbation=0.5,
                                                                max_iterations=int(max_evaluations / 100)),
                        leaders=CrowdingDistanceArchiveOmopso(100),
                        termination_criterion=StoppingByEvaluations(max=max_evaluations)
                    ),
                    algorithm_tag='OMOPSO',
                    problem_tag=problem_tag,
                    run=run,
                )
            )
            jobs.append(
                Job(
                    algorithm=GDE3(
                        problem=problem,
                        population_size=100,
                        cr=0.5,
                        f=0.5,
                        termination_criterion=StoppingByEvaluations(max=max_evaluations)
                    ),
                    algorithm_tag='GDE3',
                    problem_tag=problem_tag,
                    run=run,
                )
            )
    return jobs
