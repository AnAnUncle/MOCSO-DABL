from examples.experiment.generate_xlsx import analysis_data
from examples.experiment.settings import *

from jmetal.core.quality_indicator import *
from jmetal.lab.experiment import Experiment,  generate_summary_from_experiment_3d
from jmetal.problem import *

if __name__ == '__main__':
    # # Configure the experiments
    # problems = {'LZ09_F6': LZ09_F6}
    # 'DTLZ5': DTLZ5(), 'DTLZ6': DTLZ6(),文件比较大，运行慢
    # problems = {'DTLZ2': DTLZ2()}
    # problems = {'DTLZ2': DTLZ2(),'DTLZ4': DTLZ4(),'DTLZ7': DTLZ7()}
    problems = {'DTLZ1': DTLZ1(), 'DTLZ2': DTLZ2(), 'DTLZ4': DTLZ4(), 'DTLZ5': DTLZ5(), 'DTLZ6': DTLZ6(),
                'DTLZ7': DTLZ7(),
                "Viennet2": Viennet2()}

    n_num = 2

    # Run the study
    output_directory = 'test\\3D\\' + tagTime
    # print(output_directory)

    # jobs = configure_experiment_3d(problems=problems, n_run=n_num, max_evaluations=max_evaluations)
    jobs = configure_experiment_cso(problems=problems, n_run=n_num, max_evaluations=max_evaluations)

    # Generate summary file
    # quality_indicators = [InvertedGenerationalDistance()]
    # quality_indicators=[Spacing()]
    # quality_indicators = [InvertedGenerationalDistance(), GenerationalDistance(), EpsilonIndicator(),
    #                       HyperVolume([1.0, 1.0])]
    quality_indicators = [InvertedGenerationalDistance(), GenerationalDistance(), EpsilonIndicator(),
                          HyperVolume(reference_point=[4.0, 1.0, 6.0])]

    experiment = Experiment(output_dir=output_directory, jobs=jobs)
    experiment.run()

    tag = generate_summary_from_experiment_3d(
        input_dir=output_directory,
        reference_fronts='../../resources/reference_front',
        quality_indicators=quality_indicators
    )
    if tag:
        analysis_data()
    else:
        print("indicator calculating error")
