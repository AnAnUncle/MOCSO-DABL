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
from jmetal.core.algorithm import ChickenSwarmOptimization
from jmetal.core.problem import FloatProblem
from jmetal.core.quality_indicator import *
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

from jmetal.util.ranking import FastNonDominatedRanking, FastAggregateRanking

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
                 # mutation: Mutation,
                 leaders: Optional[BoundedArchive],
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
        # self.mutation_operator = mutation

        # ???????????????????????????
        self.mate = []
        self.mother = []
        self.cork_list = []
        self.hen_list = []
        self.chick_list = []

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
        # self.epsilon_archive = NonDominatedSolutionListArchive(EpsilonDominanceComparator(epsilon))
        self.epsilon_archive = NonDominatedSolutionListArchive(DominanceComparator())
        # ????????????????????????
        self.num = 0
        # ????????????????????????
        self.G = 0
        # ???????????????????????????
        self.alone = []

        self.mesh_div = 10
        self.MA = 800
        # self.ranking = FastNonDominatedRanking()
        self.ranking = FastNonDominatedRanking()
        self.max_eval = self.termination_criterion.max_evaluations / self.swarm_size
        # ????????????????????????IGD???
        self.igd = Que(5)
        self.igd_now = 1000000
        # ????????????????????????????????????igd?????????
        self.best_igd = 1000000
        self.best_hv = 0

        self.dominance_comparator = DominanceComparator()

    def create_initial_solutions(self) -> List[FloatSolution]:
        return [self.swarm_generator.new(self.problem) for _ in range(self.swarm_size)]

    def evaluate(self, solution_list: List[FloatSolution]):
        return self.swarm_evaluator.evaluate(solution_list, self.problem)

    def init_progress(self) -> None:
        self.evaluations = self.swarm_size

        self.initialize_particle_best(self.solutions)
        self.initialize_global_best(self.solutions)
        self.leaders.compute_density_estimator()

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def step(self):
        self.update_position(self.solutions)
        self.update_global_best(self.solutions)
        self.update_particle_best(self.solutions)
        self.leaders.compute_density_estimator()
        self.perturbation(self.solutions)
        self.chick_replacement()
        self.solutions = self.evaluate(self.solutions)
        self.update_global_best(self.solutions)
        self.update_particle_best(self.solutions)
        self.G += 1
        # self.show_size_archieve()

    def update_progress(self) -> None:
        # if self.comparator_trans_checked():
        #     self.epsilon_archive.comparator = DominanceComparator()
        # print(self.epsilon_archive.comparator)

        self.evaluations += self.swarm_size
        self.leaders.compute_density_estimator()
        # ???????????????
        self.observable_action()

    def initialize_global_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            self.leaders.add(particle)
            self.epsilon_archive.add(copy(particle))

    def initialize_particle_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            particle.attributes['local_best'] = copy(particle)

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
        MotherLib = random.sample(self.hen_list, int(len(self.hen_list)))
        mother_new = []

        for i in range(len(swarm)):
            if i in self.cork_list:
                swarm[i].attributes['label'] = 'Cork'
            if i in self.hen_list:
                swarm[i].attributes['label'] = 'Hen'
                self.mate.append([i, numpy.random.choice(self.cork_list)])
            if i in self.chick_list:
                swarm[i].attributes['label'] = 'Chick'
                mother_new.append([i, numpy.random.choice(MotherLib)])
        self.mother = mother_new

    def select_global(self) -> FloatSolution:
        # leader???????????????????????????
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

    def boundary_learn_checked(self):
        limit = self.evaluations / self.termination_criterion.max_evaluations
        if limit < 0.95:
            return False
        # print("o")
        return True

    def select_global_best(self) -> FloatSolution:
        if not self.boundary_learn_checked():
            return self.select_global()
        else:
            alone = []
            leader_list = self.leaders.solution_list
            for j in range(self.leaders.size()):
                if math.isinf(leader_list[j].attributes.get("crowding_distance")):
                    alone.append(leader_list[j])
            if len(alone) == 0:
                return self.select_global()
            else:
                # ????????????????????????????????????????????????????????????????????????
                self.alone = list(solution.objectives for solution in alone)
            if len(alone) > 1:
                particles = random.sample(alone, 1)
                best_global = copy(particles[0])
            else:
                best_global = copy(self.leaders.solution_list[0])
            return best_global

    def update_position(self, swarm: List[FloatSolution]) -> None:
        ct = 1 - self.G / self.max_eval
        ccork = 0.5 + (2.5 - 0.5) * ct

        if int(self.G) % 5 == 0:
            self.build_socity(swarm)
        # eps???????????????????????????, ??????????????????
        eps = 2.220446049250313e-16

        for i in range(self.swarm_size):
            particle = swarm[i]
            best_global = self.select_global_best()
            # ????????????
            if particle.attributes['label'] == 'Cork':
                fit1 = self.get_fitness(particle).objectives
                fit2 = best_global.objectives
                # ???tempSigma
                # if numpy.mean(fit2) > numpy.mean(fit1):
                # print("o")
                if self.dominance_comparator.compare(particle, best_global) == -1:
                    tempSigma = 1.0
                else:
                    # ??????exp????????????
                    temp = (numpy.mean(fit2) - numpy.mean(fit1)) / (numpy.abs(numpy.mean(fit1)) + eps)
                    if temp > 700:
                        tempSigma = 2.0
                    else:
                        tempSigma = numpy.exp(temp)
                normalRandom = numpy.random.normal(
                    0, tempSigma, size=particle.number_of_variables)

                for j in range(particle.number_of_variables):
                    if tempSigma == 1.0:
                        particle.variables[j] = numpy.abs(
                            numpy.sin(numpy.random.choice(numpy.linspace(-numpy.pi, numpy.pi)))) * particle.variables[
                                                    j] * (1 + normalRandom[j])
                    else:
                        particle.variables[j] = numpy.abs(
                            numpy.sin(numpy.random.choice(numpy.linspace(-numpy.pi, numpy.pi)))) * particle.variables[
                                                    j] * (1 + normalRandom[j] + ccork * (
                                best_global.variables[j] - particle.variables[j]))

                    if particle.variables[j] < self.problem.lower_bound[j]:
                        particle.variables[j] = self.problem.lower_bound[j]

                    if particle.variables[j] > self.problem.upper_bound[j]:
                        particle.variables[j] = self.problem.upper_bound[j]

            # ????????????
            if particle.attributes['label'] == 'Hen':
                r1 = round(random.uniform(self.r1_min, self.r1_max), 1)
                r2 = round(random.uniform(self.r2_min, self.r2_max), 1)
                c1 = round(random.uniform(self.c1_min, self.c1_max), 1)
                c2 = round(random.uniform(self.c2_min, self.c2_max), 1)

                best_particle = copy(swarm[i].attributes['local_best'])

                for j in range(particle.number_of_variables):
                    temp = (c1 * r1 * (best_particle.variables[j] - swarm[i].variables[j])) \
                           + (c2 * r2 * (best_global.variables[j] - swarm[i].variables[j]))
                    particle.variables[j] += temp

                    if particle.variables[j] < self.problem.lower_bound[j]:
                        particle.variables[j] = self.problem.lower_bound[j]

                    if particle.variables[j] > self.problem.upper_bound[j]:
                        particle.variables[j] = self.problem.upper_bound[j]
            # ????????????
            if particle.attributes['label'] == 'Chick':
                for j in range(particle.number_of_variables):
                    particle.variables[j] = particle.variables[j] + numpy.random.rand() * 2 * (
                            swarm[self.find_parterner(self.mother, i)].variables[j] - particle.variables[j])

                    if particle.variables[j] < self.problem.lower_bound[j]:
                        particle.variables[j] = self.problem.lower_bound[j]

                    if particle.variables[j] > self.problem.upper_bound[j]:
                        particle.variables[j] = self.problem.upper_bound[j]

    def perturbation(self, swarm: List[FloatSolution]) -> None:
        self.non_uniform_mutation.set_current_iteration(int(self.evaluations / self.swarm_size))
        for i in range(self.swarm_size):
            if (i % 3) == 0:
                self.non_uniform_mutation.execute(swarm[i])
            elif (i % 3) == 1:
                self.uniform_mutation.execute(swarm[i])

    def chick_replacing_checked(self):
        # if self.problem.number_of_objectives == 2 and len(self.epsilon_archive.solution_list) > 50:
        #     return True
        # if self.problem.number_of_objectives == 3 and len(self.epsilon_archive.solution_list) > 200:
        #     return True
        if self.evaluations > self.termination_criterion.max_evaluations / 2:
            return True
        return False

    # ????????????????????????????????????leader????????????
    def chick_replacement(self):
        if self.chick_replacing_checked():
            # print("ok")
            chicken = list(filter(self.solutions_filter, self.solutions))
            for item in chicken:
                item.variables = self.get_leader_plus().variables

    # ????????????????????????epsilon??????leader
    def get_leader_plus(self, p: int = 0.005) -> FloatSolution:
        leader_plus = self.select_global_best()
        rand = np.random.uniform(-1, 1, size=leader_plus.number_of_variables)
        for j in range(leader_plus.number_of_variables):
            leader_plus.variables[j] = leader_plus.variables[j] + rand[j] * p
            if leader_plus.variables[j] < self.problem.lower_bound[j]:
                leader_plus.variables[j] = self.problem.lower_bound[j]
            if leader_plus.variables[j] > self.problem.upper_bound[j]:
                leader_plus.variables[j] = self.problem.upper_bound[j]
        return leader_plus

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            # if self.leaders.add(copy(particle)):
            #     self.epsilon_archive.add(copy(particle))
            self.leaders.add(copy(particle))
            self.epsilon_archive.add(copy(particle))

    def update_particle_best(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            flag = self.dominance_comparator.compare(
                swarm[i],
                swarm[i].attributes['local_best'])
            if flag != 1:
                swarm[i].attributes['local_best'] = copy(swarm[i])

    def get_result(self) -> List[FloatSolution]:
        self.epsilon_archive.update()
        self.epsilon_archive.compute_density_estimator()
        front = self.epsilon_archive.solution_list
        # # ???????????????????????????????????????????????????????????????????????????????????????????????????
        # if self.problem.number_of_constraints != 0:
        #     return self.objective_recover()
        return front

    def get_name(self) -> str:
        return 'MOCSO-DABL'

    def show_size_archieve(self):
        print("archieve", len(self.leaders.solution_list))
        print("R", len(self.mother))

    # ????????????
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
        observable_data['IGD'] = len(self.leaders.non_dominated_solution_archive.solution_list)
        # observable_data['IGD'] = str(self.sp) + str(len(self.epsilon_archive.solution_list))

        # temp = self.hv_calculate()
        # if temp > self.best_hv:
        #     self.best_hv = temp
        #     self.best_result = self.epsilon_archive.solution_list
        # observable_data["HV"] = temp

        # # ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        # observable_data['SOLUTIONS'] = self.objective_recover()
        # ???????????????????????????????????????????????????observer,?????????list
        # observable_data['REFERENCE_POINT'] = self.alone
        # # ???????????????????????????
        # observable_data['SOLUTIONS'] = self.solutions
        # chicken = list(filter(self.solutions_filter, self.solutions))
        # observable_data['SOLUTIONS'] = chicken
        # ??????????????????????????????
        observable_data['SOLUTIONS'] = [self.epsilon_archive.solution_list[0]]
        # observable_data['SOLUTIONS'] = self.leaders.non_dominated_solution_archive.solution_list

        self.observable.notify_all(**observable_data)

    # ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????,???????????????????????????
    def objective_recover(self) -> List[FloatSolution]:
        if self.problem.number_of_constraints == 0:
            return self.epsilon_archive.solution_list
        result = deepcopy(self.epsilon_archive.solution_list)
        for solution in result:
            solution.objectives[0] = solution.objectives[0] * -1
        return result

    def get_index(self, obj1, swarm):
        for i in range(len(swarm)):
            if obj1.attributes['index'] == swarm[i].attributes['index']:
                return i

    def find_parterner(self, list, index):
        for i in range(len(list)):
            if index == list[i][0]:
                return list[i][1]

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
        # ???????????????????????????
        front_filename = self.create_front_filename()
        # ??????????????????
        front_objectives = get_reference_front(front_filename)
        # ????????????IGD
        result = InvertedGenerationalDistance(front_objectives).compute(self.epsilon_archive.solution_list)
        return result

    def hv_calculate(self):
        # ????????????IGD
        result = HyperVolume([4.0, 4.0]).compute(self.epsilon_archive.solution_list)
        return result

    # ???????????????????????????????????????????????????
    def solutions_filter(self, item):
        return item.attributes['label'] == "Chick"

    # ????????????igd?????????5????????????????????????
    def dif_igd(self):
        if self.igd.qsize() == self.igd.limit:
            dif = abs(self.igd.mean_value() - self.igd_now)
            def_format = dif / self.igd_now
            return def_format
        # ?????????????????????????????????
        return 100

    # ??????????????????SMPSO)
    # def perturbation(self, swarm: List[FloatSolution]) -> None:
    #     for i in range(self.swarm_size):
    #         if (i % 6) == 0:
    #             self.mutation_operator.execute(swarm[i])

    # def chick_replacing_checked(self):
    #     if self.chick_replacing == 0:
    #         return True
    #     if self.chick_replacing > 0 and self.igd_now < 0.05:
    #         self.chick_replacing -= 1
    #     return False

    # def comparator_trans_checked(self):
    #     if not self.chick_replacing_checked():
    #         return False
    #     if self.comparator_trans == 0:
    #         return True
    #     if self.comparator_trans > 0 and self.dif_igd() < 0.01:
    #         # print(self.dif_igd())
    #         self.comparator_trans -= 1
    #     return False

    # def boundary_learn_checked(self):
    #     limit = self.termination_criterion.max_evaluations - self.evaluations
    #     if limit > 2000:
    #         return False
    #     if self.comparator_trans_checked() and self.boundary_learn == 0:
    #         return True
    #     if self.boundary_learn > 0 and self.dif_igd() < 0.08:
    #         self.boundary_learn -= 1
    #         # print(limit)
    #         # print(self.boundary_learn)
    #     return False

