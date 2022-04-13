import copy
import math
import time
import random
from abc import ABC
from math import ceil
from typing import Generator, List, TypeVar
from copy import copy, deepcopy

import numpy as np

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.operator import Mutation
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution
from jmetal.operator import DifferentialEvolutionCrossover, NaryRandomSolutionSelection
from jmetal.util.aggregative_function import AggregativeFunction
from jmetal.util.constraint_handling import (
    feasibility_ratio,
    is_feasible,
    overall_constraint_violation_degree,
)
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.solutions.evaluator import Evaluator
from jmetal.util.neighborhood import WeightVectorNeighborhood
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.termination_criterion import (
    StoppingByEvaluations,
    TerminationCriterion,
)

S = TypeVar("S")
R = List[S]


class CDGMOEAD(Algorithm[FloatSolution, List[FloatSolution]], ABC):
    def __init__(
            self,
            problem: Problem,
            population_size: int,
            K: int,
            T: int,
            mutation: Mutation,
            crossover: DifferentialEvolutionCrossover,
            termination_criterion: TerminationCriterion = store.default_termination_criteria,
            population_generator: Generator = store.default_generator,
            population_evaluator: Evaluator = store.default_evaluator,
    ):
        """
        :param max_number_of_replaced_solutions: (eta in Zhang & Li paper).
        :param neighbourhood_selection_probability: Probability of mating with a solution in the neighborhood rather
               than the entire population (Delta in Zhang & Li paper).
        """
        super(CDGMOEAD, self).__init__()
        self.problem = problem
        self.population_size = population_size
        self.K = K
        self.T = T

        self.mutation_operator = mutation
        self.crossover_operator = crossover
        # self.selection_operator = selection

        self.population_generator = population_generator
        self.population_evaluator = population_evaluator

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.pareto_point_obj = None
        self.boundary_point_obj = None

    def create_initial_solutions(self) -> List[FloatSolution]:
        return [self.population_generator.new(self.problem) for _ in range(self.population_size)]

    def evaluate(self, solution_list: List[FloatSolution]):
        return self.population_evaluator.evaluate(solution_list, self.problem)

    def get_observable_data(self) -> dict:
        return {'PROBLEM': self.problem,
                'EVALUATIONS': self.evaluations,
                'SOLUTIONS': self.solutions,
                'COMPUTING_TIME': time.time() - self.start_computing_time}

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def init_progress(self) -> None:
        self.evaluations = self.population_size
        self.update_pareto_point_obj(self.solutions)
        self.boundary_point_obj = self.initial_boundary_point_obj(self.solutions)
        self.init_grid(self.solutions)

        # print(self.pareto_point_obj,self.boundary_point_obj)
        # print(list(map(lambda x: x.variables, self.solutions)))
        # print(list(map(lambda x: x.objectives, self.solutions)))
        # print('grid')
        # print(list(map(lambda x: x.attributes['grid'], self.solutions)))
        # print('neibor')
        # neibor = [x.attributes['neibor'] for x in self.solutions]
        # print([x.attributes['grid'] for x in neibor[0]])
        # print([x.attributes['grid'] for x in neibor[2]])
        # print([x.attributes['grid'] for x in neibor[2]])

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def step(self):
        self.update_pupulation(self.solutions)

        self.update_pareto_point_obj(self.solutions)
        self.update_boundary_point_obj(self.solutions)
        # print(self.pareto_point_obj)
        self.update_grid(self.solutions)

        self.population_select()

    def update_progress(self) -> None:
        self.evaluations += self.population_size

        observable_data = self.get_observable_data();
        self.observable.notify_all(**observable_data)

    def init_grid(self, solutions: List[FloatSolution]) -> None:
        for i in range(len(solutions)):
            solution = solutions[i]
            grid = [0] * solution.number_of_objectives
            solution.attributes['grid'] = np.array([])
            for j in range(solution.number_of_objectives):
                dj = (self.boundary_point_obj[j] - self.pareto_point_obj[j] + 2 * 0.9) / self.K
                grid[j] = math.ceil((solution.objectives[j] - self.pareto_point_obj[j] + 0.9) / dj)
            solution.attributes['grid'] = grid
        for i in range(len(solutions)):
            neibor = []
            for j in range(len(solutions)):
                if i != j:
                    distance = np.abs(
                        np.array(solutions[i].attributes['grid']) - np.array(solutions[j].attributes['grid']))
                    max_distance = max(distance)
                    if max_distance <= self.T:
                        neibor.append(solutions[j])
            solutions[i].attributes['neibor'] = neibor

    def update_grid(self, solutions: List[FloatSolution]) -> None:
        new_solutions = []
        solutions_del = []
        # print(len(solutions), 'inti')
        for i in range(len(solutions)):
            solution = solutions[i]
            solution_right = True
            for j in range(solution.number_of_objectives):
                if solution.objectives[j] > self.boundary_point_obj[j]:
                    solution_right = False

                    break
            if solution_right:
                new_solutions.append(solution)
            else:
                solutions_del.append(solution)
        # print(len(new_solutions),'new')
        if len(new_solutions) < self.population_size:
            solutions_add = random.sample(solutions_del, self.population_size - len(new_solutions))
            new_solutions += solutions_add
        # print(len(new_solutions), 'new2')

        self.solutions = new_solutions
        self.init_grid(self.solutions)

    def update_pupulation(self, solutions: List[FloatSolution]) -> None:
        offspring_population = []
        for i in range(len(solutions)):
            solution = solutions[i]
            if np.random.rand() < 0.9 and len(solution.attributes['neibor']) > 2:
                mating_pool = random.sample(solution.attributes['neibor'],2)
            else:
                mating_pool = random.sample(self.solutions, 2)
            mating_pool.append(solution)
            self.crossover_operator.current_individual = solution
            offspring = self.crossover_operator.execute(mating_pool)
            self.mutation_operator.execute(offspring[0])
            offspring_population.append(offspring[0])
        offspring_population = self.evaluate(offspring_population)
        # print(len(offspring_population),'pop')
        self.solutions += offspring_population

    def population_select(self) -> None:
        # print(self.pareto_point_obj,self.boundary_point_obj)
        # print(list(map(lambda x: x.objectives, self.solutions)))
        # print(list(map(lambda x: x.attributes['grid'], self.solutions)),'grid')
        if len(self.solutions) <= self.population_size:
            return
        for solution in self.solutions:
            solution.attributes['sort_order'] = [0] * self.solutions[0].number_of_objectives
        for j in range(len(self.solutions[0].objectives)):
            solutions_to_sort = []
            for k in range(self.K):
                solutions_to_sort.append([])
            for i in range(len(self.solutions)):
                grid_k = self.solutions[i].attributes['grid'][j]
                solutions_to_sort[grid_k-1].append(self.solutions[i])
            # print([x.objectives for x in solutions_to_sort[0]], 'now')
            for k in range(self.K):
                solutions_to_sort[k].sort(key=lambda x: x.objectives[j])
                # print([x.objectives for x in solutions_to_sort[k]],k,'k')
                for index, solution in enumerate(solutions_to_sort[k]):
                    solution.attributes['sort_order'][j] = index + 1
        # print(list(map(lambda x: x.attributes['sort_order'], self.solutions)))
        for solution in self.solutions:
            solution.attributes['sort_order'].sort()
        # print(list(map(lambda x: x.attributes['sort_order'], self.solutions)))
        solutions_sorted = self.radix_sort(self.solutions)
        # print(list(map(lambda x: x.attributes['sort_order'], solutions_sorted)))
        self.solutions = solutions_sorted[:self.population_size]

    def radix_sort(self, arr: List[FloatSolution]) -> List[FloatSolution]:
        n = arr[0].number_of_objectives
        for k in range(n):  # n轮排序
            bucket_list=[[] for i in range(self.population_size * 2)]
            for solution in arr:
                index = solution.attributes['sort_order'][n-1-k]
                bucket_list[index-1].append(solution)
            # 按当前桶的顺序重排列表
            arr = [j for i in bucket_list for j in i]
        return arr

    def update_pareto_point_obj(self, solutions: List[FloatSolution]) -> None:
        pareto_point_obj = copy(solutions[0].objectives)
        for j in range(solutions[0].number_of_objectives):
            min_obj = pareto_point_obj[j]
            for i in range(len(solutions)):
                min_obj = min(solutions[i].objectives[j], min_obj)
            pareto_point_obj[j] = min_obj
        self.pareto_point_obj = pareto_point_obj

    def initial_boundary_point_obj(self, solutions: List[FloatSolution]) -> List[float]:
        boundary_point_obj = copy(solutions[0].objectives)
        for j in range(solutions[0].number_of_objectives):
            max_obj = boundary_point_obj[j]
            for i in range(len(solutions)):
                max_obj = max(solutions[i].objectives[j], max_obj)
            boundary_point_obj[j] = max_obj
        return boundary_point_obj

    def update_boundary_point_obj(self, solutions: List[FloatSolution]) -> None:
        if self.evaluations % (self.population_size * 50) == 0:
            sp = []
            for i in range(len(solutions)):
                solution = solutions[i]
                for j in range(solutions[0].number_of_objectives):
                    if solution.objectives[j] < self.pareto_point_obj[j] + self.boundary_point_obj[j] / 5:
                        sp.append(solution)
            self.boundary_point_obj = self.initial_boundary_point_obj(sp)

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        new_solution = offspring_population[0]

        self.fitness_function.update(new_solution.objectives)

        new_population = self.update_current_subproblem_neighborhood(new_solution, population)

        return new_population

    def get_name(self):
        return "CDG_MOEAD"

    def get_result(self):
        return self.solutions
