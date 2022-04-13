from jmetal.problem import Psp31

if __name__ == '__main__':
    # print(num)
    # print(covariance_list)
    temp = Psp31()
    print(temp.evaluate(temp.create_solution()))
