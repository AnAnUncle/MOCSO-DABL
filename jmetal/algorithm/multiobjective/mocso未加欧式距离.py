import random
import pprint
from copy import copy
from typing import TypeVar, List, Optional

import numpy
from scipy.spatial.distance import euclidean

from jmetal.config import store
from jmetal.core.algorithm import ChickenSwarmOptimization
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.operator import UniformMutation
from jmetal.operator.mutation import NonUniformMutation
from jmetal.util.archive import BoundedArchive, NonDominatedSolutionListArchive
from jmetal.util.solutions.comparator import DominanceComparator, EpsilonDominanceComparator
from jmetal.util.solutions import Evaluator, Generator
from jmetal.util.termination_criterion import TerminationCriterion

from jmetal.util.ranking import FastNonDominatedRanking

R = TypeVar('R')

"""
.. module:: MOCSO
   :platform: Unix, Windows
   :synopsis: Implementation of MOCSO.
.. 
"""


class MOCSO(ChickenSwarmOptimization):

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 uniform_mutation: UniformMutation,
                 non_uniform_mutation: NonUniformMutation,
                 leaders: Optional[BoundedArchive],
                 epsilon: float,
                 termination_criterion: TerminationCriterion,
                 swarm_generator: Generator = store.default_generator,
                 swarm_evaluator: Evaluator = store.default_evaluator):
        """
        :param problem: The problem to solve.
        :param swarm_size: Size of the swarm.
        :param leaders: Archive for leaders.
        """
        super(MOCSO, self).__init__(
            problem=problem,
            swarm_size=swarm_size)
        self.swarm_generator = swarm_generator
        self.swarm_evaluator = swarm_evaluator

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.uniform_mutation = uniform_mutation
        self.non_uniform_mutation = non_uniform_mutation

        self.leaders = leaders

        self.epsilon = epsilon
        self.epsilon_archive = NonDominatedSolutionListArchive(EpsilonDominanceComparator(epsilon))
        self.num = 0

        self.mesh_div = 10
        self.MA = 800
        self.ranking = FastNonDominatedRanking()
        self.max_eval = self.termination_criterion.max_evaluations / self.swarm_size

        self.dominance_comparator = DominanceComparator()

    def create_initial_solutions(self) -> List[FloatSolution]:
        return [self.swarm_generator.new(self.problem) for _ in range(self.swarm_size)]

    def evaluate(self, solution_list: List[FloatSolution]):
        return self.swarm_evaluator.evaluate(solution_list, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def initialize_global_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            if self.leaders.add(particle):
                self.epsilon_archive.add(copy(particle))

    def initialize_particle_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            particle.attributes['local_best'] = copy(particle)

    # def initialize_direction(self, swarm: List[FloatSolution]) -> None:
    #     for i in range(self.swarm_size):
    #         for j in range(self.problem.number_of_variables):
    #             self.direction[i][j] = 0.0

    # def update_direction(self, swarm: List[FloatSolution],c1,c2,mate,anotherHens) -> None:
    #     for i in range(self.swarm_size):
    #         best_particle = copy(swarm[i].attributes['local_best'])
    #         best_global = self.select_global_best()
    #         if self.dominance_comparator.compare(best_global,swarm[mate])==1:
    #             best_global=swarm[mate]
    #         if self.dominance_comparator.compare(best_particle,swarm[anotherHens])==1:
    #             best_particle=swarm[anotherHens]
    #
    #         c1 = round(random.uniform(1.5, 2.0), 1)
    #         c2 = round(random.uniform(1.5, 2.0), 1)
    #         w = round(random.uniform(0.1, 0.5), 1)
    #
    #         for var in range(swarm[i].number_of_variables):
    #             self.direction[i][var] = w * self.direction[i][var] \
    #                                  + (c1 * round(numpy.random.rand(),1) * (best_particle.variables[var] - swarm[i].variables[var])) \
    #                                  + (c2 * round(numpy.random.rand(),1) * (best_global.variables[var] - swarm[i].variables[var]))

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

        self.cork_list = [info[i] for i in range(int(0.2 * self.swarm_size))]
        self.hen_list = [info[i] for i in range(int(0.2 * self.swarm_size), int(0.8 * self.swarm_size))]
        self.chick_list = [info[i] for i in range(int(0.8 * self.swarm_size), int(self.swarm_size))]
        MotherLib = random.sample(self.hen_list, int(0.5 * len(self.hen_list)))

        for i in range(len(swarm)):
            if i in self.cork_list:
                swarm[i].attributes['label'] = 'Cork'
            if i in self.hen_list:
                swarm[i].attributes['label'] = 'Hen'
                self.mate.append([i, numpy.random.choice(self.cork_list)])
            if i in self.chick_list:
                swarm[i].attributes['label'] = 'Chick'
                self.mother.append([i, numpy.random.choice(MotherLib)])
        return self.mate, self.mother, self.cork_list, self.hen_list, self.chick_list

    def find_parterner(self, list, index):
        for i in range(len(list)):
            if index == list[i][0]:
                return list[i][1]

    def get_fitness(self, solution):
        return self.problem.evaluate(solution)

    def cal_mesh_id(self, in_):
        # 计算网格编号id
        # 首先，将每个维度按照等分因子进行等分离散化，
        # 获取粒子在各维度上的编号。按照10进制将每一个维度编号等比相加,计算出值ֵ
        id_ = 0
        for i in range(self.problem.number_of_variables):
            id_dim = int(
                (in_[i] - self.problem.MINIMIZE) * self.mesh_div / (self.problem.MAXIMIZE - self.problem.MINIMIZE))
            id_ = id_ + id_dim * (self.mesh_div ** i)
        return id_

    def divide_archiving(self):
        # 对每个粒子定义网格编号
        for solution in self.epsilon_archive.solution_list:
            solution.id_archiving = self.cal_mesh_id(solution.variables)

    def get_crowd(self):
        num_ = self.epsilon_archive.size()
        self.crowd_archiving = numpy.zeros(self.epsilon_archive.size())
        index_ = [i for i in range(num_)]
        while len(index_) > 0:
            index_same = [index_[0]]  # 存放本次子循环中与index[0]粒子具有相同网格id所有检索位
            for i in range(1, len(index_)):
                if self.epsilon_archive.get(0).id_archiving == self.epsilon_archive.get(i).id_archiving:
                    index_same.append(index_[i])
            number_ = len(index_same)  # 本轮网格中粒子数
            for i in index_same:  # 更新本轮网格id下的所有粒子的拥挤度
                self.crowd_archiving[i] = number_
                index_.remove(i)  # 删除本轮网格所包含的粒子对应的索引号，避免重复计算

    def get_probability(self):

        self.divide_archiving()
        self.get_crowd()
        size = self.epsilon_archive.size()
        self.probability_archiving = numpy.zeros(size)
        for i in range(size):
            self.probability_archiving[i] = self.crowd_archiving[i] / sum(self.crowd_archiving)
        self.probability_archiving = self.probability_archiving / numpy.sum(self.probability_archiving)

    def get_clear_index(self):  # 按概率清除粒子，拥挤度高的粒子被清除的概率越高
        self.get_probability()
        size = self.epsilon_archive.size()
        len_clear = size - self.MA  # 需要清除掉的粒子数量
        clear_index = []
        while len(clear_index) < len_clear:
            random_pro = random.uniform(0.0,1.0)  # 生成0-1的随机数
            for i in range(size):
                if random_pro <= numpy.sum(self.probability_archiving[0:i + 1]):
                    if i not in clear_index:
                        clear_index.append(i)  # 记录检索值ֵ
                        break
        return clear_index

    def grid_select(self):
        clear_index = self.get_clear_index()
        for i in clear_index:
            self.epsilon_archive.delete(i)

    # def diversity(self, particle) -> None:
    #     a = numpy.random.random()
    #     for i in range(len(particle.variables)):
    #         particle.variables[i]=(particle.variables[i]-a)*numpy.exp(-(particle.variables[i]-a))
    #
    # def get_clostest(self,particle,swarm: List[FloatSolution]) -> None:
    #     diversity_list = []
    #     for i in range(len(swarm)):
    #         if swarm[i]!=particle and euclidean(particle.variables,swarm[i].variables)<=0.0075:
    #             diversity_list.append(swarm[i])
    #     return diversity_list

    # def update_position(self, swarm: List[FloatSolution]) -> None:
    #     ct = 1 - self.G / self.max_eval
    #     ccork = 0.5 + (2.5 - 0.5) * ct
    #     if int(self.G) % 10 == 0:
    #         self.mate, self.mother, self.cork_list, self.hen_list, self.chick_list = self.build_socity(swarm)
    #
    #     for i in range(self.swarm_size):
    #         particle = swarm[i]
    #         # 公鸡移动
    #         if particle.attributes['label'] == 'Cork':
    #             fit1 = self.get_fitness(particle).objectives
    #             best_global = self.select_global_best()
    #             fit2 = best_global.objectives
    #             if self.dominance_comparator.compare(particle, best_global) == 1:  # 求tempSigma
    #                 tempSigma = 1.0
    #             else:
    #                 tempSigma = numpy.exp((numpy.mean(fit2) - numpy.mean(fit1)) / (numpy.abs(numpy.mean(fit1)) + 0.001))
    #             normalRandom = numpy.random.normal(
    #                 0, tempSigma, size=particle.number_of_variables)
    #
    #             for j in range(particle.number_of_variables):
    #                 # particle.variables[j] = numpy.abs(numpy.sin(numpy.random.choice(numpy.linspace(-numpy.pi,numpy.pi))))*particle.variables[j] * (1 + normalRandom[j]+ccork*(best_global.variables[j] - particle.variables[j]))
    #                 if tempSigma == 1.0:
    #                     particle.variables[j] = numpy.abs(
    #                         numpy.sin(numpy.random.choice(numpy.linspace(-numpy.pi, numpy.pi)))) * particle.variables[
    #                                                 j] * (1 + normalRandom[j])
    #                 else:
    #                     particle.variables[j] = numpy.abs(
    #                         numpy.sin(numpy.random.choice(numpy.linspace(-numpy.pi, numpy.pi)))) * particle.variables[
    #                                                 j] * (1 + normalRandom[j] + ccork * (
    #                                 best_global.variables[j] - particle.variables[j]))
    #
    #                 if particle.variables[j] < self.problem.lower_bound[j]:
    #                     particle.variables[j] = self.problem.lower_bound[j]
    #
    #                 if particle.variables[j] > self.problem.upper_bound[j]:
    #                     particle.variables[j] = self.problem.upper_bound[j]
    #
    #         # 母鸡移动
    #         if particle.attributes['label'] == 'Hen':
    #
    #             while (True):  # 随机产生另外公鸡或母鸡的标签
    #                 anotherHens = numpy.random.choice(self.cork_list + self.hen_list)
    #                 if (anotherHens != self.find_parterner(self.mate, i)):
    #                     break
    #
    #             # 自己的适应度
    #             fit1 = numpy.mean(self.get_fitness(particle).objectives)
    #             # 所在组公鸡的适应度
    #             # fit2 = numpy.mean(self.select_global_best().objectives)
    #             fit2 = numpy.mean(self.get_fitness(swarm[self.find_parterner(self.mate, i)]).objectives)
    #             # 所在组r2的适应度
    #             fit3 = numpy.mean(self.get_fitness(swarm[anotherHens]).objectives)
    #
    #             c1 = round(numpy.exp((fit1 - fit2) / numpy.abs(fit1) + 0.00001), 1)  # 更新c1 c2
    #             c2 = round(numpy.exp(fit3 - fit1), 1)
    #
    #             # best_particle = copy(swarm[i].attributes['local_best'])        best_particle.variables[j]
    #             for m in range(len(self.mate)):
    #                 n = self.mate[m][0]
    #                 if n == i:
    #                     mate_cork = self.mate[m][1]
    #
    #             best_global = self.select_global_best()
    #
    #             for j in range(particle.number_of_variables):
    #                 particle.variables[j] += (c1 * round(numpy.random.rand(), 1) * (
    #                             swarm[mate_cork].variables[j] - swarm[i].variables[j])) \
    #                                          + (c2 * round(numpy.random.rand(), 1) * (
    #                             best_global.variables[j] - swarm[i].variables[j]))
    #
    #                 if particle.variables[j] < self.problem.lower_bound[j]:
    #                     particle.variables[j] = self.problem.lower_bound[j]
    #
    #                 if particle.variables[j] > self.problem.upper_bound[j]:
    #                     particle.variables[j] = self.problem.upper_bound[j]
    #
    #         # 小鸡移动
    #         if particle.attributes['label'] == 'Chick':
    #             for j in range(particle.number_of_variables):
    #                 particle.variables[j] = particle.variables[j] + numpy.random.rand() / 2 * numpy.random.rand() * (
    #                         swarm[self.find_parterner(self.mother, i)].variables[j] - particle.variables[j])
    #
    #                 if particle.variables[j] < self.problem.lower_bound[j]:
    #                     particle.variables[j] = self.problem.lower_bound[j]
    #
    #                 if particle.variables[j] > self.problem.upper_bound[j]:
    #                     particle.variables[j] = self.problem.upper_bound[j]
    #         if self.epsilon_archive.size() > self.MA:
    #             self.grid_select()
    #             print("ok")
    #     # print(len(self.epsilon_archive.solution_list))
    def update_position(self, swarm: List[FloatSolution]) -> None:
        ct = 1 - self.G / self.max_eval
        ccork = 0.5 + (2.5 - 0.5) * ct
        if int(self.G) % 5 == 0:
            self.mate, self.mother, self.cork_list, self.hen_list, self.chick_list = self.build_socity(swarm)

        for i in range(self.swarm_size):
            particle = swarm[i]
            # 公鸡移动
            if particle.attributes['label'] == 'Cork':
                fit1 = self.get_fitness(particle).objectives
                best_global = self.select_global_best()
                fit2 = best_global.objectives
                if self.dominance_comparator.compare(particle,best_global)==1:  # 求tempSigma
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
                                                    j] * (1 + normalRandom[j] )
                    else:
                        particle.variables[j] = numpy.abs(
                            numpy.sin(numpy.random.choice(numpy.linspace(-numpy.pi, numpy.pi)))) * particle.variables[
                                                    j] * (1 + normalRandom[j] + ccork * (best_global.variables[j] - particle.variables[j]))

                    if particle.variables[j] < self.problem.lower_bound[j]:
                        particle.variables[j] = self.problem.lower_bound[j]

                    if particle.variables[j] > self.problem.upper_bound[j]:
                        particle.variables[j] = self.problem.upper_bound[j]

            # 母鸡移动
            if particle.attributes['label'] == 'Hen':

                while (True):  # 随机产生另外公鸡或母鸡的标签
                    anotherHens = numpy.random.choice(self.cork_list+self.hen_list)
                    if(anotherHens != self.find_parterner(self.mate,i)):
                        break

                #自己的适应度
                fit1 = numpy.mean(self.get_fitness(particle).objectives)
                #所在组公鸡的适应度
                #fit2 = numpy.mean(self.select_global_best().objectives)
                fit2 = numpy.mean(self.get_fitness(swarm[self.find_parterner(self.mate,i)]).objectives)
                #所在组r2的适应度
                fit3 = numpy.mean(self.get_fitness(swarm[anotherHens]).objectives)

                c1 = round(numpy.exp((fit1-fit2)/numpy.abs(fit1)+0.00001),1)# 更新c1 c2
                c2 = round(numpy.exp(fit3-fit1), 1)

                #best_particle = copy(swarm[i].attributes['local_best'])        best_particle.variables[j]
                for m in range(len(self.mate)):
                    n = self.mate[m][0]
                    if n == i:
                        mate_cork = self.mate[m][1]
                best_particle = copy(swarm[i].attributes['local_best'])
                best_global = self.select_global_best()

                for j in range(particle.number_of_variables):
                    # //跟着公鸡学习
                    # particle.variables[j] += (c1 * round(numpy.random.rand(),1) * (swarm[mate_cork].variables[j] - swarm[i].variables[j])) \
                    #                  + (c2 * round(numpy.random.rand(),1) * (best_global.variables[j] - swarm[i].variables[j]))
                    # 跟着自身最优学习
                    particle.variables[j] += (c1 * round(numpy.random.rand(), 1) * (
                                best_particle.variables[j] - swarm[i].variables[j])) \
                                             + (c2 * round(numpy.random.rand(), 1) * (
                                best_global.variables[j] - swarm[i].variables[j]))

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
                    particle.variables[j] = particle.variables[j] + numpy.random.rand()/2 * numpy.random.rand() * (
                                swarm[self.find_parterner(self.mother, i)].variables[j] - particle.variables[j])

                    if particle.variables[j] < self.problem.lower_bound[j]:
                        particle.variables[j] = self.problem.lower_bound[j]

                    if particle.variables[j] > self.problem.upper_bound[j]:
                        particle.variables[j] = self.problem.upper_bound[j]
            if self.epsilon_archive.size() > self.MA:
                self.grid_select()

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            if self.leaders.add(copy(particle)):
                self.epsilon_archive.add(copy(particle))

    def perturbation(self, swarm: List[FloatSolution]) -> None:
        self.non_uniform_mutation.set_current_iteration(self.evaluations / self.swarm_size)
        for i in range(self.swarm_size):
            if (i % 3) == 0:
                self.non_uniform_mutation.execute(swarm[i])
            else:
                self.uniform_mutation.execute(swarm[i])

    def update_particle_best(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            flag = self.dominance_comparator.compare(
                swarm[i],
                swarm[i].attributes['local_best'])
            if flag != 1:
                swarm[i].attributes['local_best'] = copy(swarm[i])

    # def select_global_best(self) -> FloatSolution:
    #     leaders = self.leaders.solution_list
    #
    #     if len(leaders) > 2:
    #         particles = random.sample(leaders, 2)
    #
    #         if self.leaders.comparator.compare(particles[0], particles[1]) < 1:
    #             best_global = copy(particles[0])
    #         else:
    #             best_global = copy(particles[1])
    #     else:
    #         best_global = copy(self.leaders.solution_list[0])
    #
    #     return best_global

    def select_global_best(self) -> FloatSolution:
        if self.evaluations < 14000:
            return self.select_global()
        else:
            alone = []
            leader_list = self.leaders.solution_list
            for j in range(self.leaders.size()):
                if leader_list[j].attributes.get("crowding_distance") > 1:
                    alone.append(leader_list[j])
            if len(alone) == 0:
                return self.select_global()
            if len(alone) > 1:
                particles = random.sample(alone, 1)
                best_global = copy(particles[0])
                # if self.leaders.comparator.compare(particles[0], particles[1]) < 1:
                #     best_global = copy(particles[0])
                # else:
                #     best_global = copy(particles[1])

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

    def init_progress(self) -> None:
        self.evaluations = self.swarm_size
        self.leaders.compute_density_estimator()

        self.initialize_particle_best(self.solutions)
        self.initialize_global_best(self.solutions)

    def update_progress(self) -> None:
        self.evaluations += self.swarm_size
        self.leaders.compute_density_estimator()

        observable_data = self.get_observable_data()
        observable_data['SOLUTIONS'] = self.epsilon_archive.solution_list
        self.observable.notify_all(**observable_data)

    def get_result(self) -> List[FloatSolution]:
        self.epsilon_archive.update()
        self.epsilon_archive.compute_density_estimator()
        front = self.epsilon_archive.solution_list
        # # 换用leader作为最终解
        # front = self.leaders.solution_list
        # pp = pprint.PrettyPrinter(indent=4)
        # for solution in front[1:]:
        #     pp.pprint(solution)
        #     break
        return front


    def get_name(self) -> str:
        return 'AMOCSO'

    def show_size_archieve(self):
        print("archieve", len(self.epsilon_archive.solution_list))
        print("sociiety", self.num)
