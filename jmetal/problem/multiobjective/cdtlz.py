import math
from math import pi, cos, sin
import numpy
import autograd.numpy as anp
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.problem import DTLZ1, DTLZ3


class C1_DTLZ1(FloatProblem):

    def __init__(self, number_of_variables=12, number_of_objectives=3, number_of_constraints=1):
        super(C1_DTLZ1, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = number_of_constraints

        self.obj_directions = [self.MINIMIZE] * number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        k = self.number_of_variables - self.number_of_objectives + 1

        g = sum([(x - 0.5) * (x - 0.5) - cos(20.0 * pi * (x - 0.5))
                 for x in solution.variables[self.number_of_variables - k:]])

        g = 100 * (k + g)

        solution.objectives = [(1.0 + g) * 0.5] * self.number_of_objectives

        for i in range(self.number_of_objectives):
            for j in range(self.number_of_objectives - (i + 1)):
                solution.objectives[i] *= solution.variables[j]

            if i != 0:
                solution.objectives[i] *= 1 - solution.variables[self.number_of_objectives - (i + 1)]
        self.__evaluate_constraints(solution)
        return solution

    def __evaluate_constraints(self, solution: FloatSolution) -> None:
        constraints = [0.0 for _ in range(self.number_of_constraints)]
        constraints[0] = (1 - solution.objectives[-1] / 0.6 - numpy.sum([x / 0.5 for x in solution.objectives[:-1]]))
        solution.constraints = constraints

    def get_name(self):
        return 'C1_DTLZ1'


class C1_DTLZ3(FloatProblem):

    def __init__(self, number_of_variables=12, number_of_objectives=3, number_of_constraints=1, r=None):
        super(C1_DTLZ3, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = number_of_constraints

        self.obj_directions = [self.MINIMIZE] * number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

        if r is None:
            if self.number_of_objectives < 5:
                r = 9.0
            elif 5 <= self.number_of_objectives <= 12:
                r = 12.5
            else:
                r = 15.0

        self.r = r

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        k = self.number_of_variables - self.number_of_objectives + 1

        g = sum(
            [(x - 0.5) ** 2 - cos(20.0 * pi * (x - 0.5)) for x in solution.variables[self.number_of_variables - k:]])
        g = 100.0 * (k + g)

        f = [1.0 + g for _ in range(self.number_of_objectives)]

        for i in range(self.number_of_objectives):
            for j in range(self.number_of_objectives - (i + 1)):
                f[i] *= cos(solution.variables[j] * 0.5 * pi)

            if i != 0:
                aux = self.number_of_objectives - (i + 1)
                f[i] *= sin(solution.variables[aux] * 0.5 * pi)

        solution.objectives = [f[x] for x in range(self.number_of_objectives)]
        self.__evaluate_constraints(solution)
        return solution

    def __evaluate_constraints(self, solution: FloatSolution) -> None:
        constraints = [0.0 for _ in range(self.number_of_constraints)]
        radius = numpy.sum([num * num for num in solution.objectives])
        constraints[0] = (radius - 16) * (radius - self.r ** 2)
        solution.constraints = constraints

    def get_name(self):
        return 'C1_DTLZ3'


class C2_DTLZ2(FloatProblem):

    def __init__(self, number_of_variables: int = 12, number_of_objectives=3, number_of_constraints=1, r=None):
        super(C2_DTLZ2, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = number_of_constraints

        self.obj_directions = [self.MINIMIZE] * number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

        if r is None:
            if number_of_objectives == 2:
                r = 0.2
            elif number_of_objectives == 3:
                r = 0.4
            else:
                r = 0.5

        self.r = r

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        k = self.number_of_variables - self.number_of_objectives + 1

        g = sum([(x - 0.5) * (x - 0.5) for x in solution.variables[self.number_of_variables - k:]])

        solution.objectives = [1.0 + g] * self.number_of_objectives

        for i in range(self.number_of_objectives):
            for j in range(self.number_of_objectives - (i + 1)):
                solution.objectives[i] *= cos(solution.variables[j] * 0.5 * pi)

            if i != 0:
                solution.objectives[i] *= sin(0.5 * pi * solution.variables[self.number_of_objectives - (i + 1)])

        self.__evaluate_constraints(solution)
        return solution

    def __evaluate_constraints(self, solution: FloatSolution) -> None:
        constraints = [0.0 for _ in range(self.number_of_constraints)]
        obj = numpy.array([solution.objectives])
        v1 = anp.inf * anp.ones(1)

        for i in range(self.number_of_objectives):
            temp = (obj[:, i] - 1) ** 2 + (anp.sum(obj ** 2, axis=1) - obj[:, i] ** 2) - self.r ** 2
            v1 = anp.minimum(temp.flatten(), v1)

        a = 1 / anp.sqrt(self.number_of_objectives)
        v2 = anp.sum((obj - a) ** 2, axis=1) - self.r ** 2

        constraints[0] = -anp.minimum(v1, v2.flatten())[0]

        solution.constraints = constraints

    def get_name(self):
        return 'C2_DTLZ2'

class C3_DTLZ1(FloatProblem):

    def __init__(self, number_of_variables=12, number_of_objectives=3, number_of_constraints=3):
        super(C3_DTLZ1, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = number_of_constraints

        self.obj_directions = [self.MINIMIZE] * number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        k = self.number_of_variables - self.number_of_objectives + 1

        g = sum([(x - 0.5) * (x - 0.5) - cos(20.0 * pi * (x - 0.5))
                 for x in solution.variables[self.number_of_variables - k:]])

        g = 100 * (k + g)

        solution.objectives = [(1.0 + g) * 0.5] * self.number_of_objectives

        for i in range(self.number_of_objectives):
            for j in range(self.number_of_objectives - (i + 1)):
                solution.objectives[i] *= solution.variables[j]

            if i != 0:
                solution.objectives[i] *= 1 - solution.variables[self.number_of_objectives - (i + 1)]
        self.__evaluate_constraints(solution)
        return solution

    def __evaluate_constraints(self, solution: FloatSolution) -> None:
        constraints = [0.0 for _ in range(self.number_of_constraints)]
        obj = numpy.array([solution.objectives])
        for i in range(self.number_of_objectives):
            constraints[i] = -(1 - obj[:, i] / 0.5 - (anp.sum(obj, axis=1) - obj[:, i]))

        solution.constraints = constraints

    def get_name(self):
        return 'C3_DTLZ1'

class C3_DTLZ4(FloatProblem):

    def __init__(self, number_of_variables=7, number_of_objectives=3, number_of_constraints=3):
        super(C3_DTLZ4, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = number_of_constraints

        self.obj_directions = [self.MINIMIZE] * number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        k = self.number_of_variables - self.number_of_objectives + 1

        g = sum([(x - 0.5) * (x - 0.5) - cos(20.0 * pi * (x - 0.5))
                 for x in solution.variables[self.number_of_variables - k:]])

        g = 100 * (k + g)

        solution.objectives = [(1.0 + g) * 0.5] * self.number_of_objectives

        for i in range(self.number_of_objectives):
            for j in range(self.number_of_objectives - (i + 1)):
                solution.objectives[i] *= solution.variables[j]

            if i != 0:
                solution.objectives[i] *= 1 - solution.variables[self.number_of_objectives - (i + 1)]
        self.__evaluate_constraints(solution)
        return solution

    def __evaluate_constraints(self, solution: FloatSolution) -> None:
        constraints = [0.0 for _ in range(self.number_of_constraints)]
        obj = numpy.array([solution.objectives])
        for i in range(self.number_of_objectives):
            constraints[i] = -(1 - obj[:, i] / 0.5 - (anp.sum(obj, axis=1) - obj[:, i]))

        solution.constraints = constraints

    def get_name(self):
        return 'C3_DTLZ4'