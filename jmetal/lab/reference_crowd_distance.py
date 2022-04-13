import io
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from jmetal.core.algorithm import Algorithm
from jmetal.core.quality_indicator import QualityIndicator
from jmetal.util.solutions import print_function_values_to_file, print_variables_to_file, read_solutions
LOGGER = logging.getLogger('jmetal')

def generate_summary_from_experiment(input_dir: str, quality_indicators: List[QualityIndicator],
                                     reference_fronts: str = ''):
    """ Compute a list of quality indicators. The input data directory *must* met the following structure (this is generated
    automatically by the Experiment class):

    * <base_dir>

      * algorithm_a

        * problem_a

          * FUN.0.tsv
          * FUN.1.tsv
          * VAR.0.tsv
          * VAR.1.tsv
          * ...

    :param input_dir: Directory where all the input data is found (function values and variables).
    :param reference_fronts: Directory where reference fronts are found.
    :param quality_indicators: List of quality indicators to compute.
    :return: None.
    """

    if not quality_indicators:
        quality_indicators = []

    with open('QualityIndicatorSummary.csv', 'w+') as of:
        of.write('Algorithm,Problem,ExecutionId,IndicatorName,IndicatorValue\n')

    for dirname, _, filenames in os.walk(input_dir):
        for filename in filenames:
            try:
                # Linux filesystem
                algorithm, problem = dirname.split('/')[-2:]
            except ValueError:
                # Windows filesystem
                algorithm, problem = dirname.split('\\')[-2:]

            # if 'TIME' in filename:
            #     run_tag = [s for s in filename.split('.') if s.isdigit()].pop()
            #
            #     with open(os.path.join(dirname, filename), 'r') as content_file:
            #         content = content_file.read()
            #
            #     with open('QualityIndicatorSummary.csv', 'a+') as of:
            #         of.write(','.join([algorithm, problem, run_tag, 'Time', str(content)]))
            #         of.write('\n')

            if 'FUN' in filename:
                solutions = read_solutions(os.path.join(dirname, filename))
                run_tag = [s for s in filename.split('.') if s.isdigit()].pop()

                for indicator in quality_indicators:
                    reference_front_file = os.path.join(reference_fronts, problem + '.pf')
                    # reference_front_file = os.path.join(reference_fronts, problem + '.pf')

                    # Add reference front if any
                    if hasattr(indicator, 'reference_front'):
                        if Path(reference_front_file).is_file():
                            indicator.reference_front = read_solutions(reference_front_file)
                        else:
                            LOGGER.warning('Reference front not found at', reference_front_file)

                    result = indicator.compute(solutions)

                    # Save quality indicator value to file
                    with open('QualityIndicatorSummary.csv', 'a+') as of:
                        of.write(','.join([algorithm, problem, run_tag, indicator.get_name(), str(result)]))
                        of.write('\n')


def refer_crowd_distance( reference_fronts: str = '', problem: str ="DTLZ7"):
    reference_front_file = os.path.join(reference_fronts, problem + '.3D.pf')
    print(reference_front_file)
    reference_front = []
    solution = []
    temp = {}

    if Path(reference_front_file).is_file():
        reference_front = read_solutions(reference_front_file)
    length_refer = len(reference_front)
    for i in range(length_refer):
        temp["objectives"] = reference_front[i]
        solution.append(temp)
    print(solution[0]["objectives"])
# ../../resources/reference_front\DTLZ7.3D.pf



if __name__ == '__main__':
    refer_crowd_distance(reference_fronts='../../resources/reference_front')