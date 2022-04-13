import math
import numpy as np
import random
import pprint
from copy import copy
from typing import TypeVar, List, Optional

import numpy
from scipy.spatial.distance import euclidean

from jmetal.config import store
from jmetal.core.algorithm import Hybrid
from jmetal.core.problem import FloatProblem
from jmetal.core.quality_indicator import InvertedGenerationalDistance
from jmetal.core.solution import FloatSolution
from jmetal.operator import UniformMutation
from jmetal.operator.mutation import NonUniformMutation
from jmetal.util.archive import BoundedArchive, NonDominatedSolutionListArchive
from jmetal.util.distance import EuclideanDistance
from jmetal.util.queue_limited import Que
from jmetal.util.solutions.comparator import DominanceComparator, EpsilonDominanceComparator
from jmetal.util.solutions import Evaluator, Generator
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.util.get_solution import *
from jmetal.util.ranking import FastNonDominatedRanking
from typing import TypeVar, List

from jmetal.config import store
from jmetal.core.algorithm import EvolutionaryAlgorithm
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem
from jmetal.util.solutions.evaluator import Evaluator
from jmetal.util.solutions.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion
import time
from typing import TypeVar, List

import dask
from distributed import as_completed, Client

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.algorithm import DynamicAlgorithm, Algorithm
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem, DynamicProblem
from jmetal.operator import RankingAndCrowdingDistanceSelection, BinaryTournamentSelection
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.replacement import RankingAndDensityEstimatorReplacement, RemovalPolicyType
from jmetal.util.solutions import Evaluator, Generator
from jmetal.util.solutions.comparator import DominanceComparator, Comparator, RankingAndCrowdingDistanceComparator, \
    MultiComparator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: MOCSO
   :platform: Unix, Windows
   :synopsis: Implementation of MOCSO.
.. 
"""


class MOCSON(Hybrid):

    def __init__(self,
                 problem: FloatProblem,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 uniform_mutation: UniformMutation,
                 non_uniform_mutation: NonUniformMutation,
                 leaders: Optional[BoundedArchive],
                 epsilon: float,
                 selection: Selection = BinaryTournamentSelection(
                     MultiComparator([FastNonDominatedRanking.get_comparator(),
                                      CrowdingDistance.get_comparator()])),
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = store.default_comparator
                 ):
        """
        :param problem: The problem to solve.
        :param population_size: Size of the swarm.
        :param leaders: Archive for leaders.
        """
        super(MOCSON, self).__init__()
        self.problem = problem
        self.population_size = population_size
        self.offspring_population_size = offspring_population_size
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection

        self.population_generator = population_generator
        self.population_evaluator = population_evaluator

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.mating_pool_size = \
            self.offspring_population_size * \
            self.crossover_operator.get_number_of_parents() // self.crossover_operator.get_number_of_children()

        if self.mating_pool_size < self.crossover_operator.get_number_of_children():
            self.mating_pool_size = self.crossover_operator.get_number_of_children()

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.uniform_mutation = uniform_mutation
        self.non_uniform_mutation = non_uniform_mutation
        self.dominance_comparator = dominance_comparator

        self.leaders = leaders
        # CrowdingDistanceArchive(100)
        self.c1_min = 1.5
        self.c1_max = 2.0
        self.c2_min = 1.5
        self.c2_max = 2.0
        self.r1_min = 0.0
        self.r1_max = 1.0
        self.r2_min = 0.0
        self.r2_max = 1.0

        self.epsilon = epsilon
        self.epsilon_archive = NonDominatedSolutionListArchive(EpsilonDominanceComparator(epsilon))
        # 记录种群更新代数
        self.num = 0
        # 记录种群迭代次数
        self.G = 0
        # 记录边界粒子目标值
        self.alone = []

        self.mesh_div = 10
        self.MA = 800
        self.ranking = FastNonDominatedRanking()
        self.max_eval = self.termination_criterion.max_evaluations / self.population_size
        self.igd = Que(5)

    def init_progress(self) -> None:
        self.evaluations = self.population_size
        self.leaders.compute_density_estimator()

        self.initialize_particle_best(self.solutions)
        self.initialize_global_best(self.solutions)
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def step(self):
        if self.evaluations < 5000:
            mating_population = self.selection(self.solutions)
            offspring_population = self.reproduction(mating_population)
            offspring_population = self.evaluate(offspring_population)

            self.solutions = self.replacement(self.solutions, offspring_population)

        self.update_position(self.solutions)
        self.perturbation(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        self.update_global_best(self.solutions)
        self.update_particle_best(self.solutions)
        self.G += 1

    def update_progress(self) -> None:
        self.evaluations += self.offspring_population_size

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

        self.leaders.compute_density_estimator()

        observable_data = self.get_observable_data()
        observable_data['SOLUTIONS'] = self.epsilon_archive.solution_list
        # cork = list(filter(self.solutions_filter, self.solutions))
        # observable_data['SOLUTIONS'] = cork
        # 将想动态显示的点通过参考点形式传入observer,格式为list
        observable_data['REFERENCE_POINT'] = self.alone

        # temp = self.igd_calculate()
        # self.igd.put(temp)
        # observable_data['IGD'] = temp

        # self.boundary_learning(temp)
        # print(self.igd_calculate())
        self.observable.notify_all(**observable_data)

    def create_initial_solutions(self) -> List[S]:
        return [self.population_generator.new(self.problem)
                for _ in range(self.population_size)]

    def evaluate(self, population: List[S]):
        return self.population_evaluator.evaluate(population, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def selection(self, population: List[S]):
        mating_population = []

        for i in range(self.mating_pool_size):
            solution = self.selection_operator.execute(population)
            mating_population.append(solution)

        return mating_population

    def reproduction(self, mating_population: List[S]) -> List[S]:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()

        if len(mating_population) % number_of_parents_to_combine != 0:
            raise Exception('Wrong number of parents')

        offspring_population = []
        for i in range(0, self.offspring_population_size, number_of_parents_to_combine):
            parents = []
            for j in range(number_of_parents_to_combine):
                parents.append(mating_population[i + j])

            offspring = self.crossover_operator.execute(parents)

            for solution in offspring:
                self.mutation_operator.execute(solution)
                offspring_population.append(solution)
                if len(offspring_population) >= self.offspring_population_size:
                    break

        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        population.extend(offspring_population)

        population.sort(key=lambda s: s.objectives[0])

        return population[:self.population_size]

    def initialize_global_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            if self.leaders.add(particle):
                self.epsilon_archive.add(copy(particle))

    def initialize_particle_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            particle.attributes['local_best'] = copy(particle)

    def get_index(self, obj1, swarm):
        for i in range(len(swarm)):
            if obj1.attributes['index'] == swarm[i].attributes['index']:
                return i

    def build_socity(self, swarm: List[FloatSolution]):
        self.num += 1
        for i in range(len(swarm)):
            swarm[i].attributes['index'] = i
        self.mate = []
        self.mother = []
        info = []
        ranked_sublists = self.ranking.compute_ranking(swarm)

        for i in range(len(ranked_sublists)):
            for j in range(len(ranked_sublists[i])):
                info.append(self.get_index(ranked_sublists[i][j], swarm))

        self.cork_list = [info[i] for i in range(int(0.2 * self.population_size))]
        self.hen_list = [info[i] for i in range(int(0.2 * self.population_size), int(0.8 * self.population_size))]
        self.chick_list = [info[i] for i in range(int(0.8 * self.population_size), int(self.population_size))]
        MotherLib = random.sample(self.hen_list, int(0.5 * len(self.hen_list)))

        for i in range(len(swarm)):
            if i in self.cork_list:
                swarm[i].attributes['label'] = 'Cork'
                if self.evaluations > 18000:
                    best = self.select_global_best()
                    swarm[i].variables = best.variables
                    swarm[i].objectives = best.objectives
            if i in self.hen_list:
                swarm[i].attributes['label'] = 'Hen'
                self.mate.append([i, numpy.random.choice(self.cork_list)])
            if i in self.chick_list:
                swarm[i].attributes['label'] = 'Chick'
                self.mother.append([i, numpy.random.choice(MotherLib)])
        return self.mate, self.mother, self.cork_list, self.hen_list, self.chick_list

        # 过滤显示不同身份的解（目前是攻击）

    def solutions_filter(self, item):
        # filter_arr = []
        # for i in range(self.population_size):
        #     particle = self.solutions[i]
        #     if particle.attributes['label'] == tag:
        #         filter_arr.append(True)
        #     else:
        #         filter_arr.append(False)
        # return filter_arr
        return item.attributes['label'] == "Cork"

    def update_position(self, swarm: List[FloatSolution]) -> None:
        ct = 1 - self.G / self.max_eval
        ccork = 0.5 + (2.5 - 0.5) * ct
        if int(self.G) % 5 == 0:
            self.mate, self.mother, self.cork_list, self.hen_list, self.chick_list = self.build_socity(swarm)

        for i in range(self.population_size):
            particle = swarm[i]
            best_global = self.select_global_best()
            best_global2 = self.select_global_best()
            # 公鸡移动
            if particle.attributes['label'] == 'Cork':
                fit1 = self.get_fitness(particle).objectives
                fit2 = best_global.objectives
                # 求tempSigma
                # if numpy.mean(fit2) > numpy.mean(fit1):
                # print("o")
                if self.dominance_comparator.compare(particle, best_global) == -1:
                    tempSigma = 1.0
                else:
                    tempSigma = numpy.exp((numpy.mean(fit2) - numpy.mean(fit1)) / (numpy.abs(numpy.mean(fit1)) + 0.001))
                normalRandom = numpy.random.normal(
                    0, tempSigma, size=particle.number_of_variables)

                for j in range(particle.number_of_variables):
                    # 学长论文
                    # particle.variables[j] = numpy.abs(
                    #     numpy.sin(numpy.random.choice(numpy.linspace(-numpy.pi, numpy.pi)))) * particle.variables[j] * (
                    #                                     1 + normalRandom[j])
                    # 公鸡分情况
                    if tempSigma == 1.0:
                        particle.variables[j] = numpy.abs(
                            numpy.sin(numpy.random.choice(numpy.linspace(-numpy.pi, numpy.pi)))) * particle.variables[
                                                    j] * (1 + normalRandom[j])
                    else:
                        particle.variables[j] = numpy.abs(
                            numpy.sin(numpy.random.choice(numpy.linspace(-numpy.pi, numpy.pi)))) * particle.variables[
                                                    j] * (1 + normalRandom[j] + ccork * (
                                best_global.variables[j] - particle.variables[j]))

                        # if self.G < 5000:
                        #     particle.variables[j] = numpy.abs(
                        #         numpy.sin(numpy.random.choice(numpy.linspace(-numpy.pi, numpy.pi)))) * \
                        #                             particle.variables[
                        #                                 j] * (1 + normalRandom[j] + ccork * (
                        #             best_global.variables[j] - particle.variables[j]))
                        # else:
                        #     particle.variables[j] = numpy.abs(
                        #         numpy.sin(numpy.random.choice(numpy.linspace(-numpy.pi, numpy.pi)))) * \
                        #                             particle.variables[
                        #                                 j] * (1 + normalRandom[j]) + ccork * (
                        #                                     best_global.variables[j] - particle.variables[j])

                    if particle.variables[j] < self.problem.lower_bound[j]:
                        particle.variables[j] = self.problem.lower_bound[j]

                    if particle.variables[j] > self.problem.upper_bound[j]:
                        particle.variables[j] = self.problem.upper_bound[j]

            # 母鸡移动
            if particle.attributes['label'] == 'Hen':
                while (True):  # 随机产生另外公鸡或母鸡的标签
                    anotherHens = numpy.random.choice(self.cork_list + self.hen_list)
                    if (anotherHens != self.find_parterner(self.mate, i)):
                        break
                r1 = round(random.uniform(self.r1_min, self.r1_max), 1)
                r2 = round(random.uniform(self.r2_min, self.r2_max), 1)
                c1 = round(random.uniform(self.c1_min, self.c1_max), 1)
                c2 = round(random.uniform(self.c2_min, self.c2_max), 1)
                # # 自己的适应度
                # fit1 = numpy.mean(self.get_fitness(particle).objectives)
                # # 所在组公鸡的适应度
                # # fit2 = numpy.mean(self.select_global_best().objectives)
                # fit2 = numpy.mean(self.get_fitness(swarm[self.find_parterner(self.mate, i)]).objectives)
                # # 所在组r2的适应度
                # fit3 = numpy.mean(self.get_fitness(swarm[anotherHens]).objectives)
                # f1 = (fit1 - fit2) / (numpy.abs(fit1) + 0.00001)
                # # f2 = (fit1 - fit3) / (numpy.abs(fit1) + 0.00001)
                # f2 = fit3 - fit1
                # if f1 >= 700:
                #     c1 = 1
                # else:
                #     c1 = round(numpy.exp(f1), 1)
                # if f2 >= 700:
                #     c2 = 1
                # else:
                #     c2 = round(numpy.exp(f2), 1)

                # best_particle = copy(swarm[i].attributes['local_best'])        best_particle.variables[j]
                for m in range(len(self.mate)):
                    n = self.mate[m][0]
                    if n == i:
                        mate_cork = self.mate[m][1]
                best_particle = copy(swarm[i].attributes['local_best'])
                # best_global = self.select_global_best()

                for j in range(particle.number_of_variables):
                    # //跟着公鸡学习
                    # particle.variables[j] += (c1 * round(numpy.random.rand(),1) * (swarm[mate_cork].variables[j] - swarm[i].variables[j])) \
                    #                  + (c2 * round(numpy.random.rand(),1) * (best_global.variables[j] - swarm[i].variables[j]))
                    # 跟着自身最优学习
                    # temp = (c1 * round(numpy.random.rand(), 1) * (
                    #         best_particle.variables[j] - swarm[i].variables[j])) \
                    #                          + (c2 * round(numpy.random.rand(), 1) * (
                    #         best_global.variables[j] - swarm[i].variables[j]))
                    temp = (c1 * r1 * (best_particle.variables[j] - swarm[i].variables[j])) \
                           + (c2 * r2 * (best_global.variables[j] - swarm[i].variables[j]))
                    particle.variables[j] += temp
                    # print(round(temp, 5))

                    # # 跟着自身最优和公鸡学习
                    # particle.variables[j] += (c1 * round(numpy.random.rand(), 1) * (
                    #         best_particle.variables[j] - swarm[i].variables[j])) \
                    #                          + (c2 * round(numpy.random.rand(), 1) * (
                    #         swarm[mate_cork].variables[j] - swarm[i].variables[j]))
                    if particle.variables[j] < self.problem.lower_bound[j]:
                        particle.variables[j] = self.problem.lower_bound[j]

                    if particle.variables[j] > self.problem.upper_bound[j]:
                        particle.variables[j] = self.problem.upper_bound[j]
            # 小鸡移动
            if particle.attributes['label'] == 'Chick':
                for j in range(particle.number_of_variables):
                    particle.variables[j] = particle.variables[j] + numpy.random.rand() / 2 * numpy.random.rand() * (
                            swarm[self.find_parterner(self.mother, i)].variables[j] - particle.variables[j])

                    if particle.variables[j] < self.problem.lower_bound[j]:
                        particle.variables[j] = self.problem.lower_bound[j]

                    if particle.variables[j] > self.problem.upper_bound[j]:
                        particle.variables[j] = self.problem.upper_bound[j]
            if self.epsilon_archive.size() > self.MA:
                self.grid_select()

            # 公鸡移动
            # if particle.attributes['label'] == 'Cork':
            #     # 求tempSigma
            #     fit1 = self.get_fitness(particle).objectives
            #     fit2 = best_global.objectives
            #     if numpy.mean(fit2) > numpy.mean(fit1):
            #         # print("o")
            #         tempSigma = 1.0
            #     else:
            #         tempSigma = numpy.exp((numpy.mean(fit2) - numpy.mean(fit1)) / (numpy.abs(numpy.mean(fit1)) + 0.001))
            #     normalRandom = numpy.random.normal(0, tempSigma, size=particle.number_of_variables)
            #
            #     for j in range(particle.number_of_variables):
            #         adap = numpy.abs(
            #             numpy.sin(numpy.random.choice(numpy.linspace(-numpy.pi, numpy.pi))))
            #         if tempSigma == 1.0:
            #             particle.variables[j] = particle.variables[j] * (1 + normalRandom[j])
            #         else:
            #             particle.variables[j] = particle.variables[j] * (1 + normalRandom[j]) + \
            #                                     ccork * (best_global.variables[j] - particle.variables[j])
            #
            #         if particle.variables[j] < self.problem.lower_bound[j]:
            #             particle.variables[j] = self.problem.lower_bound[j]
            #
            #         if particle.variables[j] > self.problem.upper_bound[j]:
            #             particle.variables[j] = self.problem.upper_bound[j]
            #
            # # 母鸡移动
            # if particle.attributes['label'] == 'Hen':
            #     fit1 = self.get_fitness(particle).objectives
            #     fit2 = best_global.objectives
            #     fit3 = best_global2.objectives
            #
            #     for j in range(particle.number_of_variables):
            #         f1 = (numpy.mean(fit1) - numpy.mean(fit2)) / (numpy.abs(numpy.mean(fit1)) + 0.001)
            #                     f2 = numpy.mean(fit1) - numpy.mean(fit3)
            #                     if f1 >= 700:
            #                         s1 = f1
            #                     else:
            #                         s1 = numpy.exp(f1)
            #                     if f2 >= 700:
            #                         s2 = f2
            #                     else:
            #                         s2 = numpy.exp(f2)
            #         temp = (s1 * np.random.uniform(0, 1) * (best_global.variables[j] - swarm[i].variables[j])) \
            #                + (s2 * np.random.uniform(0, 1) * (best_global2.variables[j] - swarm[i].variables[j]))
            #         particle.variables[j] += temp
            #         if particle.variables[j] < self.problem.lower_bound[j]:
            #             particle.variables[j] = self.problem.lower_bound[j]
            #
            #         if particle.variables[j] > self.problem.upper_bound[j]:
            #             particle.variables[j] = self.problem.upper_bound[j]
            # # 小鸡移动
            # if particle.attributes['label'] == 'Chick':
            #     for j in range(particle.number_of_variables):
            #         particle.variables[j] = particle.variables[j] + np.random.uniform(0, 2) * (
            #                 swarm[self.find_parterner(self.mother, i)].variables[j] - particle.variables[j])
            #
            #         if particle.variables[j] < self.problem.lower_bound[j]:
            #             particle.variables[j] = self.problem.lower_bound[j]
            #
            #         if particle.variables[j] > self.problem.upper_bound[j]:
            #             particle.variables[j] = self.problem.upper_bound[j]
            # if self.epsilon_archive.size() > self.MA:
            #     self.grid_select()

    def select_global_best(self) -> FloatSolution:
        if self.evaluations < 20000:
            return self.select_global()
        else:
            alone = []
            leader_list = self.leaders.solution_list
            for j in range(self.leaders.size()):
                if math.isinf(leader_list[j].attributes.get("crowding_distance")):
                    alone.append(leader_list[j])
            # # 输出所有边界粒子目标函数值（拥挤距离为inf)
            # print(alone[0].objectives,alone[1].objectives)
            if len(alone) == 0:
                return self.select_global()
            else:
                self.alone = list(solution.objectives for solution in alone)
                # print(self.alone)
            if len(alone) > 1:
                particles = random.sample(alone, 1)
                best_global = copy(particles[0])
            else:
                best_global = copy(self.leaders.solution_list[0])
            return best_global

    def select_global(self) -> FloatSolution:
        # leader中随机选取一个粒子
        leaders = self.leaders.solution_list
        if len(leaders) > 2:
            particles = random.sample(leaders, 2)

            if self.leaders.comparator.compare(particles[0], particles[1]) < 1:
                best_global = copy(particles[0])
            else:
                best_global = copy(particles[1])
        else:
            best_global = copy(self.leaders.solution_list[0])
        return best_global

    def perturbation(self, swarm: List[FloatSolution]) -> None:
        self.non_uniform_mutation.set_current_iteration(int(self.evaluations / self.population_size))
        for i in range(self.population_size):
            if (i % 3) == 0:
                self.non_uniform_mutation.execute(swarm[i])
            else:
                self.uniform_mutation.execute(swarm[i])

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            if self.leaders.add(copy(particle)):
                self.epsilon_archive.add(copy(particle))

    def update_particle_best(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.population_size):
            flag = self.dominance_comparator.compare(
                swarm[i],
                swarm[i].attributes['local_best'])
            if flag != 1:
                swarm[i].attributes['local_best'] = copy(swarm[i])

    def create_front_filename(self) -> str:
        if self.problem.number_of_objectives == 2:
            return "../../resources/reference_front/" + self.problem.get_name() + ".pf"
        elif self.problem.number_of_objectives == 3:
            return "../../resources/reference_front/" + self.problem.get_name() + "." + \
                   str(self.problem.number_of_objectives) + "D.pf"
        else:
            return "problem dimenson error"

    def igd_calculate(self):
        # 获取最优前沿文件名
        front_filename = self.create_front_filename()
        # 获取最优前沿
        front_objectives = get_reference_front(front_filename)
        # 计算当前IGD
        result = InvertedGenerationalDistance(front_objectives).compute(self.epsilon_archive.solution_list)
        return result

    def boundary_learning(self, now_igd: float):
        dif = abs(self.igd.mean_value() - now_igd)
        if self.igd.qsize() == self.igd.limit and dif < now_igd / 100:
            print("")
            print(self.igd.queue)
            print(dif)
            print(now_igd / 100, now_igd)
            print("exit")

    # 有问题
    def find_parterner(self, list, index):
        for i in range(len(list)):
            if index == list[i][0]:
                return list[i][1]
        a = self.population_size
        return np.random.randint(0, a)

    def get_fitness(self, solution):
        return self.problem.evaluate(solution)

    def get_result(self) -> List[FloatSolution]:
        self.epsilon_archive.update()
        self.epsilon_archive.compute_density_estimator()
        # front = self.epsilon_archive.solution_list
        front = self.epsilon_archive.solution_list
        # pp = pprint.PrettyPrinter(indent=4)
        # for solution in front[1:]:
        #     pp.pprint(solution)
        #     break
        return front

    def get_name(self) -> str:
        return 'MOCSON'

    def show_size_archieve(self):
        print("archieve", len(self.epsilon_archive.solution_list))
        print("sociiety", self.num)
