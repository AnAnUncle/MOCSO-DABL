import math
import traceback

import numpy as np
import random
import pprint
from copy import copy, deepcopy
from typing import TypeVar, List, Optional

import numpy
from scipy.spatial.distance import euclidean

from jmetal.config import store
from jmetal.core.operator import Mutation
from jmetal.core.algorithm import ChickenSwarmOptimization19
from jmetal.core.problem import FloatProblem
from jmetal.core.quality_indicator import *
from jmetal.core.solution import FloatSolution
from jmetal.operator import UniformMutation
from jmetal.operator.mutation import NonUniformMutation
from jmetal.util.archive_mocso_19 import BoundedArchive, NonDominatedSolutionListArchive
from jmetal.util.distance import EuclideanDistance
from jmetal.util.queue_limited import Que
from jmetal.util.solutions.comparator import DominanceComparator, EpsilonDominanceComparator
from jmetal.util.solutions import Evaluator, Generator
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.util.get_solution import *

from jmetal.util.ranking import FastNonDominatedRanking, FastAggregateRanking

R = TypeVar('R')

"""
.. module:: MOCSO
   :platform: Unix, Windows
   :synopsis: Implementation of MOCSO.
.. 
"""


class MOCSO19(ChickenSwarmOptimization19):

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 archive: Optional[BoundedArchive],
                 termination_criterion: TerminationCriterion,
                 swarm_generator: Generator = store.default_generator,
                 swarm_evaluator: Evaluator = store.default_evaluator):
        """
        :param problem: The problem to solve.
        :param swarm_size: Size of the swarm.
        :param archive: Archive
        """
        super(MOCSO19, self).__init__(
            problem=problem,
            swarm_size=swarm_size)
        self.swarm_generator = swarm_generator
        self.swarm_evaluator = swarm_evaluator

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        # 创建鸡群不同等级群
        self.mate = []
        self.mother = []
        self.cork_list = []
        # 用来保存种群中的公鸡
        self.rooster = []
        self.hen_list = []
        self.chick_list = []

        self.archive = archive
        # 记录种群更新代数
        self.num = 0
        # 记录种群迭代次数
        self.G = 0
        # 记录边界粒子目标值
        self.alone = []

        self.mesh_div = 10
        self.MA = 800
        # self.ranking = FastNonDominatedRanking()
        self.ranking = FastAggregateRanking()
        self.max_eval = self.termination_criterion.max_evaluations / self.swarm_size
        # 定长队列记录实习IGD值
        self.igd = Que(5)
        self.igd_now = 1000000
        # 用来存储最优粒子群和最优igd，淘汰
        self.best_igd = 1000000
        self.best_hv = 0

        self.dominance_comparator = DominanceComparator()

    def create_initial_solutions(self) -> List[FloatSolution]:
        return [self.swarm_generator.new(self.problem) for _ in range(self.swarm_size)]

    def evaluate(self, solution_list: List[FloatSolution]):
        return self.swarm_evaluator.evaluate(solution_list, self.problem)

    def init_progress(self) -> None:
        self.evaluations = self.swarm_size

        self.initialize_archive(self.solutions)
        self.archive.compute_density_estimator()

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def step(self):
        self.update_position(self.solutions)
        self.improve_diversity(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        self.update_archive(self.solutions)
        self.G += 1
        # self.show_size_archieve()

    def update_progress(self) -> None:
        self.evaluations += self.swarm_size
        # 监听器配置
        self.observable_action()

    def initialize_archive(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            self.archive.add(particle)

    def select_rooster(self) -> FloatSolution:
        # return random.choice(self.archive.solution_list)
        return random.choice(self.rooster)

    def build_socity(self, swarm: List[FloatSolution]):
        self.num += 1
        for i in range(len(swarm)):
            swarm[i].attributes['index'] = i
        info = []
        ranked_sublists = self.ranking.compute_ranking(swarm)

        for i in range(len(ranked_sublists)):
            for j in range(len(ranked_sublists[i])):
                info.append(self.get_index(ranked_sublists[i][j], swarm))

        self.cork_list = [info[i] for i in range(int(0.2 * self.swarm_size))]
        self.hen_list = [info[i] for i in range(int(0.2 * self.swarm_size), int(0.8 * self.swarm_size))]
        self.chick_list = [info[i] for i in range(int(0.8 * self.swarm_size), int(self.swarm_size))]
        mother_lib = random.sample(self.hen_list, int(len(self.hen_list)))
        mother_new = []
        rooster_new = []
        for i in range(len(swarm)):
            if i in self.cork_list:
                swarm[i].attributes['label'] = 'Cork'
                rooster_new.append(swarm[i])
            if i in self.hen_list:
                swarm[i].attributes['label'] = 'Hen'
                self.mate.append([i, numpy.random.choice(self.cork_list)])
            if i in self.chick_list:
                swarm[i].attributes['label'] = 'Chick'
                mother_new.append([i, numpy.random.choice(mother_lib)])
            self.rooster = rooster_new
            self.mother = mother_new

    def update_position(self, swarm: List[FloatSolution]) -> None:
        if int(self.G) % 5 == 0:
            self.build_socity(swarm)
        # eps是一个很小的非负数, 避免分母为零
        eps = 2.220446049250313e-16

        for i in range(self.swarm_size):
            particle = swarm[i]

            # 公鸡移动
            if particle.attributes['label'] == 'Cork':
                xk = self.select_rooster()
                fit1 = particle.objectives
                fit2 = xk.objectives
                if self.dominance_comparator.compare(particle, xk) == -1:
                    tempSigma = 1.0
                else:
                    # 加入exp溢出避免
                    temp = (numpy.mean(fit2) - numpy.mean(fit1)) / (numpy.abs(numpy.mean(fit1)) + eps)
                    if temp > 700:
                        tempSigma = 2.0
                    else:
                        tempSigma = numpy.exp(temp)

                for j in range(particle.number_of_variables):
                    normalRandom = numpy.random.normal(
                        0, tempSigma, size=1)[0]
                    particle.variables[j] = particle.variables[
                                                j] * (1 + normalRandom)
                    if particle.variables[j] < self.problem.lower_bound[j]:
                        particle.variables[j] = self.problem.lower_bound[j]

                    if particle.variables[j] > self.problem.upper_bound[j]:
                        particle.variables[j] = self.problem.upper_bound[j]

            # 母鸡移动
            if particle.attributes['label'] == 'Hen':
                xr1 = self.select_rooster()
                xr2 = self.select_rooster()
                r1 = numpy.random.rand()
                r2 = numpy.random.rand()

                fit_x = particle.objectives
                fit_r1 = xr1.objectives
                fit_r2 = xr2.objectives
                exponential1 = (numpy.mean(fit_r1) - numpy.mean(fit_x)) / (numpy.abs(numpy.mean(fit_x)) + eps)
                exponential2 = numpy.mean(fit_r2) - numpy.mean(fit_x)
                # 防溢出
                c1 = numpy.exp(exponential1 if exponential1 < 700 else 700)
                c2 = numpy.exp(exponential2 if exponential2 < 700 else 700)

                for j in range(particle.number_of_variables):
                    temp = (c1 * r1 * (xr1.variables[j] - particle.variables[j])) \
                           + (c2 * r2 * (xr2.variables[j] - particle.variables[j]))
                    particle.variables[j] += temp

                    if particle.variables[j] < self.problem.lower_bound[j]:
                        particle.variables[j] = self.problem.lower_bound[j]

                    if particle.variables[j] > self.problem.upper_bound[j]:
                        particle.variables[j] = self.problem.upper_bound[j]
            # 小鸡移动
            if particle.attributes['label'] == 'Chick':
                for j in range(particle.number_of_variables):
                    fl = numpy.random.rand() * 0.4 + 0.5
                    particle.variables[j] += fl * (
                            swarm[self.find_parterner(self.mother, i)].variables[j] - particle.variables[j])

                    if particle.variables[j] < self.problem.lower_bound[j]:
                        particle.variables[j] = self.problem.lower_bound[j]

                    if particle.variables[j] > self.problem.upper_bound[j]:
                        particle.variables[j] = self.problem.upper_bound[j]

    def improve_diversity(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            particle = swarm[i]
            for j in range(self.swarm_size):
                if i != j:
                    neibor = swarm[j]
                    for k in range(particle.number_of_variables):
                        dif = np.abs(particle.variables[k] - neibor.variables[k])
                        if dif <= 0.0075:
                            rand = random.uniform(self.problem.lower_bound[k], self.problem.upper_bound[k])
                            squre = np.square(self.problem.upper_bound[k] - self.problem.lower_bound[k])
                            step = rand / squre
                            dif = particle.variables[k] - step
                            # 防溢出
                            coefficient = numpy.exp(-dif) / squre if dif < 10 else 0

                            particle.variables[k] = dif * coefficient
                            if particle.variables[k] < self.problem.lower_bound[k]:
                                particle.variables[k] = self.problem.lower_bound[k]
                            if particle.variables[k] > self.problem.upper_bound[k]:
                                particle.variables[k] = self.problem.upper_bound[k]

    def update_archive(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            self.archive.add(copy(particle))

    def get_result(self) -> List[FloatSolution]:
        front = self.archive.solution_list
        # # 应用问题为约束问题，检测有约束就对取负后的目标函数恢复之后进行输出
        # if self.problem.number_of_constraints != 0:
        #     return self.objective_recover()
        return front

    def get_index(self, obj1, swarm):
        for i in range(len(swarm)):
            if obj1.attributes['index'] == swarm[i].attributes['index']:
                return i

    def find_parterner(self, list, index):
        for i in range(len(list)):
            if index == list[i][0]:
                return list[i][1]

    def get_name(self) -> str:
        return 'DAMOCSO'

    def show_size_archieve(self):
        print("archieve", len(self.archive.solution_list))
        print("R", len(self.rooster))

    # 工具函数
    def observable_action(self):
        observable_data = self.get_observable_data()
        # temp = self.igd_calculate()
        # # if temp < self.best_igd:
        # #     self.best_igd = temp
        # #     # self.best_result = self.epsilon_archive.solution_list
        # # self.igd.put(temp)
        # # self.igd_now = temp
        # observable_data['IGD'] = temp
        # observable_data['IGD'] = self.dif_igd()
        # observable_data['IGD'] = self.best_igd
        # observable_data['IGD'] = len(self.epsilon_archive.solution_list)
        # observable_data['IGD'] = len(self.archive.non_dominated_solution_archive.solution_list)
        # observable_data['IGD'] = str(self.sp) + str(len(self.epsilon_archive.solution_list))

        # temp = self.hv_calculate()
        # if temp > self.best_hv:
        #     self.best_hv = temp
        #     self.best_result = self.epsilon_archive.solution_list
        # observable_data["HV"] = temp

        # # 最大化函数恢复后动态显示（深拷贝修复后结果，不影响之前值）；普通问题，正常输出精英集
        # observable_data['SOLUTIONS'] = self.objective_recover()
        # 将想动态显示的点通过参考点形式传入observer,格式为list
        # observable_data['REFERENCE_POINT'] = self.alone
        # # 输出全部解动态显示
        # observable_data['SOLUTIONS'] = self.solutions
        # chicken = list(filter(self.solutions_filter, self.solutions))
        # observable_data['SOLUTIONS'] = chicken
        # 输出精英集备选输出解
        # observable_data['SOLUTIONS'] = self.epsilon_archive.solution_list
        observable_data['SOLUTIONS'] = self.archive.non_dominated_solution_archive.solution_list

        self.observable.notify_all(**observable_data)

    # 对于普通问题，输出精英集；对于约束投资组合优化问题，将第一个改造后最小化的目标函数恢复为最大化,返回拷贝更新后的值
    def objective_recover(self) -> List[FloatSolution]:
        if self.problem.number_of_constraints == 0:
            return self.archive.solution_list
        result = deepcopy(self.archive.solution_list)
        for solution in result:
            solution.objectives[0] = solution.objectives[0] * -1
        return result

    def get_fitness(self, solution):
        return self.problem.evaluate(solution)

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
        result = InvertedGenerationalDistance(front_objectives).compute(self.archive.solution_list)
        return result

    def hv_calculate(self):
        # 计算当前IGD
        result = HyperVolume([4.0, 4.0]).compute(self.archive.solution_list)
        return result

    # 过滤显示不同身份的解（目前是公鸡）
    def solutions_filter(self, item):
        return item.attributes['label'] == "Chick"

    # 计算当前igd与最近5次迭代平均值差异
    def dif_igd(self):
        if self.igd.qsize() == self.igd.limit:
            dif = abs(self.igd.mean_value() - self.igd_now)
            def_format = dif / self.igd_now
            return def_format
        # 不符合则返回足够大的数
        return 100
