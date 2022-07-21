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

    mat0 = [[0.42, 0.15, 0.4], [0.72, 0.1, 0.54], [0.01, 0.19, 0.42], [0.3, 0.35, 0.68]]
    mat1 = [[0.21, 0.67, 0.14, 0.96, 0.87], [0.87, 0.42, 0.2, 0.32, 0.89], [0.03, 0.56, 0.8, 0.69, 0.09]]
    mat2 = [[0.04, 0.87, 0.42, 0.53], [0.17, 0.1, 0.95, 0.69]]

    weight.append(mat0)
    weight.append(mat1)
    weight.append(mat2)

    sum = 0
    for i in range(0, len(weight), 1):

        for j in range(0, len(weight[i]), 1):
            for k in range(0, len(weight[i][0]), 1):
                if k != 0:
                    sum = sum + weight[i][j][k] * weight[i][j][k]

    sum = lamda / (2 * len(train)) * sum

    final = [[0.75, 0.75], [0.98, 0.28]]

    a = []
    res = [[1, 1], [0.32, 0.83], [0.68, 0.02]]
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

    cost = 0
    matt = a[len(a) - 1]

    for i in range(0, len(train), 1):
        J = 0
        for j in range(0, 2, 1):
            J = J - final[j][i] * np.log(matt[j][i])
            J = J - (1 - final[j][i]) * np.log(1 - matt[j][i])

        print("------")
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


train = [[0.32, 0.68], [0.83, 0.02]]
gradient, cost = trainning(train, 2, 0.25)
print(gradient)
print(cost)
