# 导入需要使用的包
import numpy
import xlrd  # 读取Excel文件的包
import xlsxwriter  # 将文件写入Excel的包


# 打开一个excel文件
def open_xls(file):
    f = xlrd.open_workbook(file)
    return f


# 获取excel中所有的sheet表
def getsheet(f):
    return f.sheets()


# 获取sheet表的行数
def get_Allrows(f, sheet):
    table = f.sheets()[sheet]
    return table.nrows


# 读取文件内容并返回行内容
def getFile(file, shnum):
    datavalue = []
    f = open_xls(file)
    table = f.sheets()[shnum]
    num = table.nrows
    for row in range(num):
        rdata = table.row_values(row)
        datavalue.append(rdata)
    return datavalue


# 获取sheet表的个数
def getshnum(f):
    x = 0
    sh = getsheet(f)
    for sheet in sh:
        x += 1
    return x


def write_xlsx(rvalue, outputxls):
    wb = xlsxwriter.Workbook(outputxls)
    # 创建一个sheet工作对象
    ws = wb.add_worksheet()
    for a in range(len(rvalue)):
        for b in range(len(rvalue[a])):
            try:
                c = float(rvalue[a][b])
            except ValueError:
                c = rvalue[a][b]
            ws.write(a, b, c)
    wb.close()


def format_array(array):
    for item in array:
        item[-1] = ""

# 函数入口，GD指标去除
def merge(allxls, outputxls):
    # 定义要合并的excel文件列表
    # 存储所有读取的结果
    rvalue = []
    for fl in allxls:
        f = open_xls(fl)
        x = getshnum(f)
        tags = "".join(fl.split('/')[-1:])[:-5].split("-")[0:2]
        for shnum in range(x):
            # print("正在读取文件：" + str(fl) + "的第" + str(shnum) + "个sheet表的内容...")
            content_array = getFile(fl, shnum)
            # 抽取表各项的关键字如std,IGD
            for item in content_array:
                # print(tags)
                item.extend(tags)
            rvalue.extend(content_array)
    # 过滤掉Problem项,DTLZ1,DTLZ3
    algorithms = rvalue[0][1:]
    algorithms[-1] = ""
    # print(algorithms)
    rvalue = list(filter(lambda n: n[0] != "Problem", rvalue))
    # 四个数组用来保存最终的数据,初始化
    IGD = [["AIGD"]]
    IGD[0].extend(algorithms)
    # print(IGD)
    GD = [["AGD"]]
    GD[0].extend(algorithms)
    HV = [["AHV"]]
    HV[0].extend(algorithms)
    EP = [["AEP"]]
    EP[0].extend(algorithms)
    for item in rvalue:
        flag = item[-1]
        # print(flag)
        if flag == "IGD":
            IGD.append(item)
        if flag == "GD":
            GD.append(item)
        if flag == "HV":
            HV.append(item)
        if flag == "EP":
            EP.append(item)
    IGD.sort(key=lambda n: n[0])
    GD.sort(key=lambda n: n[0])
    HV.sort(key=lambda n: n[0])
    EP.sort(key=lambda n: n[0])
    format_array(IGD)
    format_array(GD)
    format_array(HV)
    format_array(EP)
    # # print(IGD)
    # try:
    #     # Linux filesystem
    #     c = float("nihao")
    # except ValueError:
    #     # Windows filesystem
    #     c = float("23")
    # print(c)
    # result_content = numpy.hstack((IGD, GD, HV, EP))
    result_content = numpy.hstack((IGD, HV, EP))
    write_xlsx(result_content, outputxls)
    # wb = xlsxwriter.Workbook(outputxls)
    # # 创建一个sheet工作对象
    # ws = wb.add_worksheet()
    # for a in range(len(rvalue)):
    #     for b in range(len(rvalue[a])):
    #         c = rvalue[a][b]
    #         ws.write(a, b, c)
    # wb.close()

    print("文件合并完成")
# # 函数入口
# def merge(allxls, outputxls):
#     # 定义要合并的excel文件列表
#     # 存储所有读取的结果
#     rvalue = []
#     for fl in allxls:
#         f = open_xls(fl)
#         x = getshnum(f)
#         tags = "".join(fl.split('/')[-1:])[:-5].split("-")[0:2]
#         for shnum in range(x):
#             # print("正在读取文件：" + str(fl) + "的第" + str(shnum) + "个sheet表的内容...")
#             content_array = getFile(fl, shnum)
#             # 抽取表各项的关键字如std,IGD
#             for item in content_array:
#                 # print(tags)
#                 item.extend(tags)
#             rvalue.extend(content_array)
#     # 过滤掉Problem项,DTLZ1,DTLZ3
#     algorithms = rvalue[0][1:]
#     algorithms[-1] = ""
#     # print(algorithms)
#     rvalue = list(filter(lambda n: n[0] != "Problem", rvalue))
#     # 四个数组用来保存最终的数据,初始化
#     IGD = [["AIGD"]]
#     IGD[0].extend(algorithms)
#     # print(IGD)
#     GD = [["AGD"]]
#     GD[0].extend(algorithms)
#     HV = [["AHV"]]
#     HV[0].extend(algorithms)
#     EP = [["AEP"]]
#     EP[0].extend(algorithms)
#     for item in rvalue:
#         flag = item[-1]
#         # print(flag)
#         if flag == "IGD":
#             IGD.append(item)
#         if flag == "GD":
#             GD.append(item)
#         if flag == "HV":
#             HV.append(item)
#         if flag == "EP":
#             EP.append(item)
#     IGD.sort(key=lambda n: n[0])
#     GD.sort(key=lambda n: n[0])
#     HV.sort(key=lambda n: n[0])
#     EP.sort(key=lambda n: n[0])
#     format_array(IGD)
#     format_array(GD)
#     format_array(HV)
#     format_array(EP)
#     # # print(IGD)
#     # try:
#     #     # Linux filesystem
#     #     c = float("nihao")
#     # except ValueError:
#     #     # Windows filesystem
#     #     c = float("23")
#     # print(c)
#     result_content = numpy.hstack((IGD, GD, HV, EP))
#     write_xlsx(result_content, outputxls)
#     # wb = xlsxwriter.Workbook(outputxls)
#     # # 创建一个sheet工作对象
#     # ws = wb.add_worksheet()
#     # for a in range(len(rvalue)):
#     #     for b in range(len(rvalue[a])):
#     #         c = rvalue[a][b]
#     #         ws.write(a, b, c)
#     # wb.close()
#
#     print("文件合并完成")


if __name__ == '__main__':
    merge(["./latex/statistical/Std-IGD.xlsx", "./latex/statistical/Std-GD.xlsx"], "./latex/statistical/test.xlsx")
