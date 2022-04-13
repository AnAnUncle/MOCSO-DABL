from shutil import copyfile

from examples.experiment.generate_xlsx import analysis_data
from jmetal.core.quality_indicator import *
from jmetal.lab.experiment import Experiment, generate_summary_from_experiment, generate_summary_from_experiment_3d

if __name__ == '__main__':
    tagTime = "2022-03-28-116-20000all_test(fix_cso)1.0"
    output_directory = 'data/3D/' + tagTime
    # print(output_directory)

    # Generate summary file
    # quality_indicators = [InvertedGenerationalDistance()]
    # quality_indicators=[Spacing()]
    # quality_indicators = [InvertedGenerationalDistance(), GenerationalDistance(), EpsilonIndicator(),
    #                       HyperVolume([1.0, 1.0])]
    quality_indicators = [InvertedGenerationalDistance(), EpsilonIndicator(), HyperVolume(reference_point=[4.0, 1.0, 6.0])]
    # quality_indicators = [InvertedGenerationalDistance(), GenerationalDistance(), EpsilonIndicator(),
    #                       HyperVolume(reference_point=[4.0, 1.0, 6.0])]
    tag = generate_summary_from_experiment_3d(
        input_dir=output_directory,
        reference_fronts='../../resources/reference_front',
        quality_indicators=quality_indicators
    )
    # tag = True
    if tag:
        # 备份csv文件
        outputName = "history/indicator/3D/" + tagTime + "_boxplots.csv"
        copyfile("QualityIndicatorSummary.csv", outputName)

        analysis_data()
        # 备份xslx文件
        outputName2 = "history/xlsx/3D/" + tagTime + "_boxplots.xlsx"
        copyfile("latex/statistical/aresult.xlsx", outputName2)
    else:
        print("indicator calculating error")