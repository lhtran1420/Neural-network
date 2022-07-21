import numpy
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import StratifiedKFold

data = genfromtxt('datasets/hw3_house_votes_84.csv', delimiter=',', skip_header=1)

y = []

for i in range(0, 435, 1):
    y.append(data[i][16])

skf = StratifiedKFold(n_splits=10)
StratifiedKFold(n_splits=2, random_state=None, shuffle=False)


def trainning(train, layer, lamda):
    weight = []

    mat0 = [[0.4, 0.1], [0.3, 0.2]]
    mat1 = [[0.7, 0.5, 0.6]]
    weight.append(mat0)
    weight.append(mat1)

    final = [[0.9, 0.23]]

    arr = []
    for i in range(0, len(train), 1):
        arr.append(1)

    a = []
    res = [[1, 1], [0.13, 0.42]]
    a.append(res)

    for i in range(0, len(weight), 1):

        result = np.matmul(weight[i], a[i])

        if i < len(weight) - 1:
            arr1 = []
            for j in range(0, len(train), 1):
                arr1.append(1)
            # them 1 vao row dau
            mat = 1 / (1 + np.exp(-result))

            x1 = np.array(arr1)
            x2 = np.array(mat)
            result = np.vstack([x1, x2])
        else:
            mat = 1 / (1 + np.exp(-result))
            result = mat

        a.append(result)

    fix = []
    for i in range(len(a) - 1, 0, -1):
        if i == len(a) - 1:
            fix.append(np.subtract(a[i], final))

        else:
            x = np.array(a[i])

            matrix = np.multiply(np.matmul(numpy.transpose(weight[i]), fix[len(fix) - 1]), np.multiply(x, 1 - x))
            res2 = np.delete(matrix, 0, 0)
            fix.append(res2)

    sum = 0
    for i in range(0, len(weight), 1):

        for j in range(0, len(weight[i]), 1):
            for k in range(0, len(weight[i][0]), 1):
                if k != 0:
                    sum = sum + weight[i][j][k] * weight[i][j][k]

    sum = lamda / (2 * len(train)) * sum

    cost = 0
    matt = a[len(a) - 1]

    for i in range(0, len(train), 1):
        J = 0
        for j in range(0, 1, 1):
            J = J - final[j][i] * np.log(matt[j][i])
            J = J - (1 - final[j][i]) * np.log(1 - matt[j][i])

        print("--------+++++")
        print(J)
        cost = cost + J

    cost1 = cost / len(train)

    gradient = []
    theta = []
    for i in range(len(fix) - 1, -1, -1):
        theta.append(fix[i])

    for i in range(0, layer + 1, 1):
        mat = np.matmul(theta[i], numpy.transpose(a[i]))
        w = weight[i]

        mt = []
        for j in range(0, len(w), 1):
            arr = []
            for k in range(0, len(w[0]), 1):
                if k == 0:
                    arr.append(0)
                else:
                    arr.append(w[j][k])
            mt.append(arr)

        mat = 1 / len(train) * (mat + lamda * np.array(mt))
        gradient.append(mat)

    return gradient, cost1 + sum


train = [0.13, 0.42]
gradient, cost = trainning(train, 1, 0)
print(gradient)
print(cost)
