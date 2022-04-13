from shutil import copyfile

from examples.experiment.generate_xlsx import analysis_data
from examples.experiment.settings import *

from jmetal.core.quality_indicator import *
from jmetal.lab.experiment import Experiment, generate_summary_from_experiment, generate_summary_from_experiment_psp
from jmetal.problem import *
from jmetal.problem.multiobjective.lircmop import *

if __name__ == '__main__':

    # Configure the experiments
    # problems = {'Schaffer': Schaffer(), 'Kursawe': Kursawe(),'Fonseca': Fonseca(),
    #             'ZDT1': ZDT1(), 'ZDT2': ZDT2(), 'ZDT3': ZDT3(), 'ZDT4': ZDT4(), 'ZDT6': ZDT6()}
    # problems = {'LZ09_F1': LZ09_F1(), 'LZ09_F2': LZ09_F2(), 'LZ09_F3': LZ09_F3(), 'LZ09_F4': LZ09_F4(),
    #             'LZ09_F5': LZ09_F5(), 'LZ09_F7': LZ09_F7(), 'LZ09_F8': LZ09_F8(), 'LZ09_F9': LZ09_F9(), }
    # problems = {'LIRCMOP1': LIRCMOP1(), 'LIRCMOP2': LIRCMOP2(), 'LIRCMOP3': LIRCMOP3(), 'LIRCMOP4': LIRCMOP4(),
    #             'LIRCMOP5': LIRCMOP5(), 'LIRCMOP6': LIRCMOP6(), 'LIRCMOP7': LIRCMOP7(), 'LIRCMOP8': LIRCMOP8(),
    #             'LIRCMOP9': LIRCMOP9(), 'LIRCMOP10': LIRCMOP10(), 'LIRCMOP11': LIRCMOP11(), 'LIRCMOP12': LIRCMOP12()
    #             }
    problems = {'Psp31': Psp31(), 'Psp85': Psp85(), 'Psp89': Psp89(), 'Psp98': Psp98(), 'Psp225': Psp225()}

    n_num = 30

    # Run the study
    output_directory = 'data\\2D\\' + "2021-03-14-00-20000投资组合优化问题测试（1.0）"
    # print(output_directory)

    jobs = configure_experiment_psp(problems=problems, n_run=n_num, max_evaluations=max_evaluations)
    # jobs = configure_experiment_cso(problems=problems, n_run=n_num, max_evaluations=max_evaluations)

    # Generate summary file
    # quality_indicators = [InvertedGenerationalDistance()]
    # quality_indicators=[Spacing()]
    quality_indicators = [InvertedGenerationalDistance(), EpsilonIndicator(),
                          HyperVolume([1, 1])]
    # quality_indicators = [InvertedGenerationalDistance(), EpsilonIndicator(),
    #                       HyperVolume([4.0, 4.0])]
    # quality_indicators = [InvertedGenerationalDistance(), GenerationalDistance(), EpsilonIndicator(),
    #                       HyperVolume([4.0, 4.0])]

    experiment = Experiment(output_dir=output_directory, jobs=jobs)
    # experiment.run()

    tag = generate_summary_from_experiment_psp(
        input_dir=output_directory,
        reference_fronts='../../resources/reference_front',
        quality_indicators=quality_indicators
    )
    if tag:
        # 备份csv文件
        outputName = "history/indicator/2D/" + tagTime + ".csv"
        copyfile("QualityIndicatorSummary.csv", outputName)
        analysis_data()
        # 备份xslx文件
        outputName2 = "history/xlsx/2D/" + tagTime + ".xlsx"
        copyfile("latex/statistical/aresult.xlsx", outputName2)
    else:
        print("indicator calculating error")
