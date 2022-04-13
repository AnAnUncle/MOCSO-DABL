import math
import os


def numb(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def get_format_data(filename: str):
    contents = []
    path = os.path.join(PROJECT_ROOT, "po/" + filename)  # 文件路径
    with open(path) as f:
        string = f.readlines()
        for item in string:
            contents.append(item.strip().split(" "))
    contents = [[numb(item2) for item2 in item1 if item2 != ""] for item1 in contents]
    n = int(contents[0][0])
    return n, contents[1:n + 1], contents[n + 1:]


# 读取原始数据
# 输入portf1-f5文件地址
# def get_feference_data(filename: str):
#     contents = []
#     path = os.path.join(PROJECT_ROOT, filename)
#     # 文件路径
#     with open(path) as f:
#         string = f.readlines()
#         for item in string:
#             vector = item.strip().split(" ")
#             vector = [float(item2) for item2 in vector if item2 != ""]
#             solution = FloatSolution([], [], 2)
#             solution.number_of_objectives = 2
#             solution.objectives = vector
#             contents.append(solution)
#     return contents[1:]


# def plot_psp_reference_data():
#     for i in range(5):
#         tag = "f" + str(i + 1)
#         filename = "po/porte" + tag + ".txt"
#         reference_data_now = get_feference_data(filename)
#         plot_front = Plot(plot_title='Pareto front approximation. Problem: ', axis_labels=["profit", "risk"])
#         plot_front.plot_output(reference_data_now, label=tag, filename="image/port/" + tag, tag=tag)


def format_i_j(i: int, j: int):
    a = i
    b = j
    if i > j:
        a = j
        b = i
    return a, b


# def get_covariance_list():
#     result = [[get_covariance(i + 1, j + 1) * profit[i][1] * profit[j][1] for j in range(num)] for i in range(num)]
#     return result


# 输入行号列好，输出协方差
def get_covariance(covariance_triangle: list, num: int, i: int, j: int):
    row, col = format_i_j(i, j)
    index = int((row - 1) * (2 * num - row + 2) / 2 + col - row)
    result = covariance_triangle[index]
    # print(TriplesToSparse(covariance_triangle))
    if i == result[0] and j == result[1] or i == result[1] and j == result[0]:
        # print(result)
        # print(index)
        return result[2]
    else:
        print("error_get_covariance" + str(i) + "and" + str(j))
        print(result)
        print(index)
        return 0


def get_num_profit_covariance(filename: str):
    num, profit, covariance_triangle = get_format_data(filename)
    covariance_list = [
        [get_covariance(covariance_triangle, num, i + 1, j + 1) * profit[i][1] * profit[j][1] for j in range(num)] for i
        in range(num)]
    return num, profit, covariance_list


PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))  # 获取项目根目录
# num, profit, covariance_triangle = get_format_data()

# reference_data = get_feference_data("po/portef1.txt")
if __name__ == '__main__':
    a, b, c = get_num_profit_covariance("port5.txt")
    # print(a)
    # print(b)
    print(c)
    # print(profit)
    # print(covariance_triangle)
    # plot_psp_reference_data()

    # test
    # for i in range(num):
    #     for j in range(num):
    #         print(get_covariance(i+1, j+1) == covariance_list[i][j])
    # print(get_covariance(31, 31))

    # reference_front = read_solutions(filename='FUN.5.tsv')
    # plot_front = Plot(plot_title='Pareto front approximation. Problem: ',
    #                   reference_front=reference_data, axis_labels=["x", "y"])
    # plot_front.plot(reference_data, label="test", filename="image/f1", tag="test2")

    # plot_front = Plot(plot_title='Pareto front approximation. Problem: ')
    # plot_front.plot(reference_front, label="nihao", filename="image/a")
