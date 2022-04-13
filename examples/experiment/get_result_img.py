import os
from pathlib import Path

from jmetal.lab.visualization import Plot
from jmetal.util.solutions import read_solutions, read_solutions_3d
from jmetal.util.solutions.helper import read_port_reference


def get_objective_points_img_3d(filename: str, output: str):
    reference_fronts = '../../resources/reference_front'
    for dirname, _, filenames in os.walk(filename):
        for filename in filenames:
            # print(str(filename))
            try:
                # Linux filesystem
                algorithm, problem = dirname.split('/')[-2:]
            except ValueError:
                # Windows filesystem
                algorithm, problem = dirname.split('\\')[-2:]

            # print(run_tag == "0")
            run_tag = [s for s in filename.split('.') if s.isdigit()].pop()
            tag = '_'.join([algorithm, problem, run_tag])
            # if run_tag != "0":
            #     break

            if 'FUN.0' in filename:
                solutions = read_solutions_3d(os.path.join(dirname, filename))
                dim = len(solutions[0].objectives)
                reference_front_file = ""
                reference_front = []
                if dim == 3:
                    reference_front_file = os.path.join(reference_fronts, problem + '.3D.pf')
                elif dim == 2:
                    reference_front_file = os.path.join(reference_fronts, problem + '.pf')
                else:
                    print("dimension error")
                if Path(reference_front_file).is_file():
                    reference_front = read_solutions_3d(reference_front_file)
                plot_front = Plot(plot_title='Pareto front approximation. Problem: ', reference_front=reference_front,
                                  axis_labels=["f1", "f2", "f3"])
                plot_front.plot_output(solutions, label=algorithm,
                                       filename="image/total_algorithm/" + output + "/" + tag,
                                       tag=algorithm)


def get_objective_points_img_2d(filename: str, output: str):
    reference_fronts = '../../resources/reference_front'
    for dirname, _, filenames in os.walk(filename):
        for filename in filenames:
            try:
                # Linux filesystem
                algorithm, problem = dirname.split('/')[-2:]
            except ValueError:
                # Windows filesystem
                algorithm, problem = dirname.split('\\')[-2:]
            run_tag = [s for s in filename.split('.') if s.isdigit()].pop()
            tag = '_'.join([algorithm, problem, run_tag])
            # if run_tag != "0":
            #     break

            if 'FUN.0' in filename:
                solutions = read_solutions(os.path.join(dirname, filename))
                dim = len(solutions[0].objectives)
                reference_front_file = ""
                reference_front = []
                if dim == 3:
                    reference_front_file = os.path.join(reference_fronts, problem + '.3D.pf')
                elif dim == 2:
                    reference_front_file = os.path.join(reference_fronts, problem + '.pf')
                else:
                    print("dimension error")
                if Path(reference_front_file).is_file():
                    reference_front = read_solutions(reference_front_file)

                # solution = FloatSolution([], [], 2)
                # solution.objectives = [0.5, 0.5]
                # solution.number_of_variables = 2
                plot_front = Plot(plot_title='Pareto front approximation. Problem: ', reference_front=reference_front,
                                  axis_labels=["f1", "f2"])
                plot_front.plot_output(solutions, label=algorithm,
                                       filename="image/total_algorithm/" + output + "/" + tag,
                                       tag=algorithm)


def get_objective_points_img_psp(filename: str):
    reference_fronts = '../../resources/reference_front'
    for dirname, _, filenames in os.walk(filename):
        for filename in filenames:
            # print(str(filename))
            try:
                # Linux filesystem
                algorithm, problem = dirname.split('/')[-2:]
            except ValueError:
                # Windows filesystem
                algorithm, problem = dirname.split('\\')[-2:]
            # print(run_tag == "0")
            run_tag = [s for s in filename.split('.') if s.isdigit()].pop()
            tag = '_'.join([algorithm, problem, run_tag])
            if run_tag != "0":
                break

            if 'FUN' in filename:
                solutions = read_solutions(os.path.join(dirname, filename))
                dim = len(solutions[0].objectives)
                reference_front_file = ""
                reference_front = []
                if dim == 3:
                    reference_front_file = os.path.join(reference_fronts, problem + '.3D.pf')
                elif dim == 2:
                    reference_front_file = os.path.join(reference_fronts, problem + '.txt')
                else:
                    print("dimension error")
                if Path(reference_front_file).is_file():
                    reference_front = read_solutions(reference_front_file)

                # solution = FloatSolution([], [], 2)
                # solution.objectives = [0.5, 0.5]
                # solution.number_of_variables = 2
                plot_front = Plot(plot_title='Pareto front approximation. Problem: ', reference_front=reference_front,
                                  axis_labels=["f1", "f2", "f3"])
                plot_front.plot_output(solutions, label=algorithm, filename="image/psp/" + tag,
                                       tag=algorithm)


def plot_psp_reference_data():
    PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))  # 获取项目根目录
    for i in range(5):
        tag = "portef" + str(i + 1)
        filename = "po/" + tag + ".txt"
        path = os.path.join(PROJECT_ROOT, filename)
        reference_data_now = read_port_reference(path)
        plot_front = Plot(plot_title='Pareto front approximation. Problem: ', axis_labels=["profit", "risk"])
        plot_front.plot_output(reference_data_now, label=tag, filename="image/port/" + tag, tag=tag)


if __name__ == '__main__':
    # 生成pareto逼近情况图所需要的数据文件地址
    # tagTime = "2021-03-04-04-20000超参去除改造DAMOCSO（3.0)测试"

    # 投资组合优化问题pareto逼近情况图生成
    # filename_2d = 'data\\2D\\' + tagTime
    # get_objective_points_img_psp(filename_2d)

    # 二目标pareto逼近情况图生成
    tagTime = "2022-03-30-2232-20000all_test_constrain2.0"
    filename_2d = 'data/2D/' + tagTime
    get_objective_points_img_2d(filename_2d, '2D-constrain')

    tagTime = "2022-03-28-115-20000all_test(fix_cso)1.0"
    filename_2d = 'data/2D/' + tagTime
    get_objective_points_img_2d(filename_2d,'2D')

    # 三目标pareto逼近情况图生成
    tagTime = "2022-03-31-1917-20000all_test_constrain3d2.0"
    filename_3d = 'data/3D/' + tagTime
    get_objective_points_img_3d(filename_3d,'3D-constrain')

    tagTime = "2022-03-28-116-20000all_test(fix_cso)1.0"
    filename_3d = 'data/3D/' + tagTime
    get_objective_points_img_3d(filename_3d,'3D')

    # 投资组合优化问题数据集pareto前沿生成
    # plot_psp_reference_data()
