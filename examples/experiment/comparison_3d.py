from shutil import copyfile

from examples.experiment.generate_xlsx import analysis_data, analysis_data_3d
from examples.experiment.settings import *

from jmetal.core.quality_indicator import *
from jmetal.lab.experiment import Experiment, generate_summary_from_experiment_3d
from jmetal.problem import *

if __name__ == '__main__':
    # problems = {'DTLZ1': DTLZ1(), 'DTLZ2': DTLZ2(),'DTLZ3': DTLZ3(), 'DTLZ4': DTLZ4(), 'DTLZ5': DTLZ5(), 'DTLZ6': DTLZ6(),
    #             'DTLZ7': DTLZ7(),
    #             "Viennet2": Viennet2()}
    problems = {'DTLZ1': DTLZ1(),'DTLZ2': DTLZ2(),'DTLZ3': DTLZ3(),'DTLZ4': DTLZ4(), 'DTLZ5': DTLZ5(),
                'DTLZ6': DTLZ6(),
                'DTLZ7': DTLZ7(),
                "Viennet2": Viennet2()}

    n_num = 30

    # Run the study
    output_directory = 'data/3D/' + tagTime
    # print(output_directory)

    jobs = configure_experiment_3d(problems=problems, n_run=n_num, max_evaluations=max_evaluations)
    # jobs = configure_experiment_cso(problems=problems, n_run=n_num, max_evaluations=max_evaluations)
    # jobs = configure_experiment_cso19(problems=problems, n_run=n_num, max_evaluations=max_evaluations)

    # Generate summary file
    # quality_indicators = [InvertedGenerationalDistance()]
    # quality_indicators=[Spacing()]
    quality_indicators = [InvertedGenerationalDistance(), EpsilonIndicator(),
                          HyperVolume(reference_point=[4.0, 1.0, 6.0])]
    # quality_indicators = [InvertedGenerationalDistance(), GenerationalDistance(), EpsilonIndicator(),
    #                       HyperVolume(reference_point=[4.0, 1.0, 6.0])]

    experiment = Experiment(output_dir=output_directory, jobs=jobs)
    experiment.run()

    tag = generate_summary_from_experiment_3d(
        input_dir=output_directory,
        reference_fronts='../../resources/reference_front',
        quality_indicators=quality_indicators
    )

    if tag:
        # 备份csv文件
        outputName = "history/indicator/3D/" + tagTime + ".csv"
        copyfile("QualityIndicatorSummary_3d.csv", outputName)
        analysis_data_3d()
        # 备份xslx文件
        outputName2 = "history/xlsx/3D/" + tagTime + ".xlsx"
        copyfile("latex/statistical/aresult.xlsx", outputName2)
    else:
        print("indicator calculating error")
