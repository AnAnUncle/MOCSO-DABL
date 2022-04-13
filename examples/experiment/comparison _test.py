from examples.experiment.generate_xlsx import analysis_data
from examples.experiment.settings import *

from jmetal.core.quality_indicator import *
from jmetal.lab.experiment import Experiment, generate_summary_from_experiment
from jmetal.problem import *
from jmetal.problem.multiobjective.lircmop import *

if __name__ == '__main__':
    # Configure the experiments
    # problems = {'ZDT1': ZDT1()}
    # problems = {'Schaffer': Schaffer()}
    # problems = {'ZDT1': ZDT1(), 'ZDT2': ZDT2(), 'ZDT3': ZDT3(), 'ZDT4': ZDT4(), 'ZDT6': ZDT6()}
    # problems = {'LIRCMOP1': LIRCMOP1(), 'LIRCMOP2': LIRCMOP2()}
    # problems = {'Tanaka': Tanaka()}
    # 'ZDT1': ZDT1(), 'ZDT2': ZDT2(), 'ZDT3': ZDT3(), 'ZDT4': ZDT4(), 'ZDT6': ZDT6(),
    # 'LZ09_F1': LZ09_F1(), 'LZ09_F2': LZ09_F2(), 'LZ09_F3': LZ09_F3(), 'LZ09_F4': LZ09_F4(),
    # 'LZ09_F5': LZ09_F5(), 'LZ09_F7': LZ09_F7(), 'LZ09_F8': LZ09_F8(), 'LZ09_F9': LZ09_F9(),
    # 'LIRCMOP1': LIRCMOP1(),

    # problems = {'LIRCMOP2': LIRCMOP2(), 'Schaffer': Schaffer(), 'Kursawe': Kursawe(),
    #             'Srinivas': Srinivas(),'Tanaka': Tanaka()}
    # problems = {'Schaffer': Schaffer(), 'Kursawe': Kursawe(),'Fonseca': Fonseca(),
    #             'ZDT1': ZDT1(), 'ZDT2': ZDT2(), 'ZDT3': ZDT3(), 'ZDT4': ZDT4(), 'ZDT6': ZDT6()}
    problems = {'Psp31': Psp31(), 'Psp85': Psp85(), 'Psp89': Psp89(), 'Psp98': Psp98(), 'Psp225': Psp225()}
    n_num = 2

    # Run the study
    output_directory = 'test\\2D\\' + tagTime
    # print(output_directory)

    jobs = configure_experiment_2d(problems=problems, n_run=n_num, max_evaluations=max_evaluations)
    # jobs = configure_experiment_cso(problems=problems, n_run=n_num, max_evaluations=max_evaluations)

    # Generate summary file
    # quality_indicators = [InvertedGenerationalDistance()]
    # quality_indicators=[Spacing()]
    quality_indicators = [InvertedGenerationalDistance(), EpsilonIndicator(),
                          HyperVolume([0, 0.01])]
    # quality_indicators = [InvertedGenerationalDistance(), GenerationalDistance(), EpsilonIndicator(),
    #                       HyperVolume([4.0, 4.0])]

    experiment = Experiment(output_dir=output_directory, jobs=jobs)
    experiment.run()

    tag = generate_summary_from_experiment(
        input_dir=output_directory,
        reference_fronts='../../resources/reference_front',
        quality_indicators=quality_indicators
    )
    if tag:
        analysis_data()
    else:
        print("indicator calculating error")
