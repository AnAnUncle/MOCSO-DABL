from shutil import copyfile

from examples.experiment.generate_xlsx import analysis_data
from jmetal.core.quality_indicator import *
from jmetal.lab.experiment import Experiment, generate_summary_from_experiment
from jmetal.problem import *

if __name__ == '__main__':
    tagTime = "2022-03-28-115-20000all_test(fix_cso)1.0"
    output_directory = 'data/2D/' + tagTime
    # print(output_directory)

    # Generate summary file
    # quality_indicators = [InvertedGenerationalDistance()]
    # quality_indicators=[Spacing()]
    # quality_indicators = [InvertedGenerationalDistance(), GenerationalDistance(), EpsilonIndicator(),
    #                       HyperVolume([1.0, 1.0])]
    # quality_indicators = [EpsilonIndicator()]
    # quality_indicators = [InvertedGenerationalDistance(), GenerationalDistance(), EpsilonIndicator(),
    #                       HyperVolume([220, 50])]
    # test_problem = Tanaka()
    # solution = FloatSolution([1],[1],2)
    # solution.variables=[1.0E-4,0.03674848699046586]
    # print(test_problem.evaluate((solution)).constraints)
    quality_indicators = [InvertedGenerationalDistance(), EpsilonIndicator(),
                          HyperVolume([4.0, 4.0])]
    tag = generate_summary_from_experiment(
        input_dir=output_directory,
        reference_fronts='../../resources/reference_front',
        quality_indicators=quality_indicators
    )
    tag = True
    if tag:
        # 备份csv文件
        outputName = "history/indicator/2D/" + tagTime + "_boxplots.csv"
        copyfile("QualityIndicatorSummary.csv", outputName)

        analysis_data()
        # 备份xslx文件
        outputName2 = "history/xlsx/2D/" + tagTime + "_boxplots.xlsx"
        copyfile("latex/statistical/aresult.xlsx", outputName2)
    else:
        print("indicator calculating error")