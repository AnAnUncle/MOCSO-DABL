from typing import List

from jmetal.core.algorithm import S
from jmetal.util.solutions import read_solutions
import pandas as pd
import numpy as np


def get_reference_front(reference_front: str = '') -> List[S]:
    # print(reference_front)
    solutions = read_solutions(reference_front)
    if solutions is None:
        raise Exception('Front is none!')

    reference_front_points = pd.DataFrame(list(solution for solution in solutions))
    return [front for front in np.array(reference_front_points).flatten()]


if __name__ == '__main__':
    print(get_reference_front(reference_front='../../resources/reference_front/DTLZ7.3D.pf'))
