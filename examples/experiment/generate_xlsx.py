import csv
import xlwt
from examples.experiment.merge import merge
from jmetal.core.solution import FloatSolution
from jmetal.lab.experiment import generate_latex_tables
from typing import TypeVar, List, Optional


def csv_to_xlsx(csvfile, outfile):
    with open(csvfile, encoding='utf-8') as fc:
        r_csv = csv.reader(fc)
        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet('sheet1')  # 创建一个sheet表格
        i = 0
        for line in r_csv:
            line = str(line[0]).split()
            j = 0
            for v in line:
                try:
                    v = float((float(v)))
                    sheet.write(i, j, v)
                except BaseException:
                    sheet.write(i, j, v)
                j = j + 1
            i = i + 1
        workbook.save(outfile)  # 保存Excel


def generate_filename(math=[], quality=[], expand="", dir_name="./latex/statistical/"):
    temp = []
    for i in range(len(math)):
        for j in range(len(quality)):
            temp.append(dir_name + math[i] + "-" + quality[j] + expand)
    return temp


def generate_filename_end(math=[], expand="", dir_name="./latex/statistical/"):
    temp = []
    for i in range(len(math)):
        temp.append(dir_name + math[i] + expand)
    return temp


def ctox_array(filenamein=[], filenameout=[]):
    lenin = len(filenamein)
    lenout = len(filenameout)
    if lenin == lenout:
        for i in range(len(filenamein)):
            csv_to_xlsx(filenamein[i], filenameout[i])
    else:
        print("error lens of in and out")


# allxls = ["./latex/statistical/Mean-IGD.xlsx", "./latex/statistical/Mean-GD.xlsx","./latex/statistical/Mean-HV.xlsx", "./latex/statistical/Mean-EP.xlsx"]

# outputxls = "./latex/statistical/ALL.xlsx"
# file = './latex/statistical/Mean-IGD.csv'  # 待转化的源文件
# outfile = './latex/statistical/Mean-IGD.xlsx'  # 转化后的excel所处的位置与文件名

# 只计算Mean，std,去除GD
def analysis_data():
    generate_latex_tables(filename='QualityIndicatorSummary.csv')
    # generate_latex_tables(filename='QualityIndicatorSummary_3d.csv')
    # math = ["IQR", "Mean", "Median", "Std", "Best", "Worst"]
    # quality_p = ["IGD", "GD", "HV", "EP"]
    math = ["Mean", "Std", "Best", "Worst"]
    quality_p = ["IGD", "HV", "EP"]
    # csv转xlsx
    filenamein = generate_filename(math=math, quality=quality_p, expand=".csv")
    filenameout = generate_filename(math=math, quality=quality_p, expand=".xlsx")
    ctox_array(filenamein, filenameout)
    # ctox_array("./latex/")
    # 合并xlsx
    allxls = generate_filename(math=math, quality=quality_p, expand=".xlsx")
    outputxls_result = "./latex/statistical/" + "aresult.xlsx"
    merge(allxls, outputxls_result)


def analysis_data_3d():
    generate_latex_tables(filename='QualityIndicatorSummary_3d.csv')
    # generate_latex_tables(filename='QualityIndicatorSummary_3d.csv')
    # math = ["IQR", "Mean", "Median", "Std", "Best", "Worst"]
    # quality_p = ["IGD", "GD", "HV", "EP"]
    math = ["Mean", "Std", "Best", "Worst"]
    quality_p = ["IGD", "HV", "EP"]
    # csv转xlsx
    filenamein = generate_filename(math=math, quality=quality_p, expand=".csv")
    filenameout = generate_filename(math=math, quality=quality_p, expand=".xlsx")
    ctox_array(filenamein, filenameout)
    # ctox_array("./latex/")
    # 合并xlsx
    allxls = generate_filename(math=math, quality=quality_p, expand=".xlsx")
    outputxls_result = "./latex/statistical/" + "aresult.xlsx"
    merge(allxls, outputxls_result)


if __name__ == '__main__':
    analysis_data()
