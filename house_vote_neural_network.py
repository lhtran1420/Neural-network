import random

import matplotlib.pyplot as plt
import numpy
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data = genfromtxt('datasets/hw3_house_votes_84.csv', delimiter=',', skip_header=1)

y = []

for i in range(0, 435, 1):
    y.append(data[i][16])

skf = StratifiedKFold(n_splits=10)
StratifiedKFold(n_splits=2, random_state=None, shuffle=False)


def check(test, weight):
    mt = []
    res = []

    arr1 = []
    arr1.append(1)
    mt.append(arr1)

    for i in range(0, 16, 1):
        arr = []
        arr.append(data[test][i])
        mt.append(arr)

    for i in range(0, len(weight), 1):
        res = np.matmul(weight[i], mt)
        if i < len(weight) - 1:
            arr1 = []
            for j in range(0, len(res[0]), 1):
                arr1.append(1)

            mt0 = np.array(arr1)
            mt1 = np.array(1 / (1 + np.exp(-res)))
            mt = np.vstack([mt0, mt1])

    # print(res)

    if res[0][0] >= res[1][0]:
        return 0
    return 1


def trainning(train, layer, list_neural, lamda):
    weight = []
    last = 0

    for i in range(0, layer, 1):
        mat = []
        if i == 0:
            num_neural = random.choice(list_neural)
            last = num_neural
            for j in range(0, num_neural, 1):
                arr = []
                for k in range(0, 17, 1):
                    arr.append(random.uniform(-1, 1))
                mat.append(arr)
            last = last + 1

        else:
            num_neural = random.choice(list_neural)
            for j in range(0, num_neural, 1):
                arr = []
                for k in range(0, last, 1):
                    arr.append(random.uniform(-1, 1))
                mat.append(arr)
            last = num_neural + 1

        weight.append(mat)

    mat1 = []
    for i in range(0, 2, 1):
        arr = []
        for k in range(0, last, 1):
            arr.append(random.uniform(-1, 1))
        mat1.append(arr)

    weight.append(mat1)

    final = []
    for i in range(0, 2, 1):
        arr = []
        for j in range(0, len(train), 1):
            if data[train[j]][16] == i:
                arr.append(1)
            else:
                arr.append(0)

        final.append(arr)

    for cnt in range(0, 1000, 1):
        # print("cnt-----")
        # print(cnt)
        res = []
        a = []

        for i in range(0, 17, 1):
            arr = []

            for j in range(0, len(train), 1):
                if i == 0:
                    arr.append(1)
                else:
                    arr.append(data[train[j]][i - 1])
            res.append(arr)

        a.append(res)

        # print(a)
        for i in range(0, len(weight), 1):

            result = np.matmul(weight[i], a[i])
            if i < len(weight) - 1:
                arr1 = []
                for j in range(0, len(res[0]), 1):
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

        gradient = []

        theta = []
        for i in range(len(fix) - 1, -1, -1):
            theta.append(fix[i])

        for i in range(0, layer + 1, 1):
            mat = np.matmul(theta[i], numpy.transpose(a[i]))
            mat = 1 / len(train) * (mat + lamda * np.array(weight[i]))
            gradient.append(mat)

        alpha = 0.01
        for i in range(0, layer + 1, 1):
            weight[i] = weight[i] - alpha * gradient[i]
    return weight


def performance(train, test, layer, list_neural, lamda):
    weight = trainning(train, layer, list_neural, lamda)

    true_pos0, true_pos1 = 0, 0
    false_pos0, false_pos1 = 0, 0
    pos0, pos1 = 0, 0

    for i in range(0, len(test), 1):

        result = check(test[i], weight)

        if data[test[i]][16] == 1:
            pos1 += 1
            if result == 1:
                true_pos1 += 1
            else:
                false_pos0 += 1


        elif data[test[i]][16] == 0:
            pos0 += 1
            if result == 0:
                true_pos0 += 1
            else:
                false_pos1 += 1

    accura = (true_pos1 + true_pos0) / len(test)
    # print(accura)

    precision = 0
    if true_pos1 + false_pos1 > 0:
        precision += true_pos1 / (true_pos1 + false_pos1)

    if true_pos0 + false_pos0 > 0:
        precision += true_pos0 / (true_pos0 + false_pos0)

    precision = precision / 2

    recall = (true_pos1 / pos1 + true_pos0 / pos0) / 2

    F1 = 2 * (precision * recall) / (precision + recall)
    # print(F1)
    # print("done")
    return accura, F1


def costFunction(train):
    # print("len-----")
    # print(len(weight))
    # print(weight)
    weight = trainning(train, 2, [60, 61, 62, 63], 1)

    final = []
    for i in range(0, 2, 1):
        arr = []
        for j in range(0, len(train), 1):
            if data[train[j]][16] == i:
                arr.append(1)
            else:
                arr.append(0)

        final.append(arr)

    result = []
    for i in range(0, 17, 1):
        arr = []

        for j in range(0, len(train), 1):
            if i == 0:
                arr.append(1)
            else:
                arr.append(data[train[j]][i - 1])
        result.append(arr)

    for i in range(0, len(weight), 1):
        result = np.matmul(weight[i], result)
        if i < len(weight) - 1:
            arr1 = []
            for j in range(0, len(result[0]), 1):
                arr1.append(1)
            # them 1 vao row dau
            mat = 1 / (1 + np.exp(-result))

            x1 = np.array(arr1)
            x2 = np.array(mat)
            result = np.vstack([x1, x2])
        else:
            mat = 1 / (1 + np.exp(-result))
            result = mat

    cost = 0
    for i in range(0, len(train), 1):
        J = 0
        for j in range(0, 2, 1):
            if final[j][i] == 0:
                J += -np.log(1 - result[j][i])
            else:
                J += -np.log(result[j][i])

        cost = cost + J
    cost = cost / len(train)

    sum = 0
    for i in range(0, len(weight), 1):
        for j in range(0, len(weight[i]), 1):
            for k in range(0, len(weight[i][0]), 1):
                if k != 0:
                    sum = sum + weight[i][j][k] * weight[i][j][k]

    lamda = 1
    sum = lamda / (2 * len(train)) * sum
    return cost + sum


accura1, F1 = 0.0, 0.0
accura1_1, F1_1 = 0.0, 0.0
accura2, F2 = 0.0, 0.0
accura3, F3 = 0.0, 0.0
accura4, F4 = 0.0, 0.0
accura5, F5 = 0.0, 0.0
accura6, F6 = 0.0, 0.0

turn = 0

for train_index, test_index in skf.split(data, y):
    list_neural1 = [20, 21, 22, 23]
    lamda1 = 0.2
    accuracy1, F1_score1 = performance(train_index, test_index, 1, list_neural1, lamda1)
    accura1 = accura1 + accuracy1
    F1 = F1 + F1_score1

    lamda2 = 1
    accuracy1_1, F1_score1_1 = performance(train_index, test_index, 1, list_neural1, lamda2)
    accura1_1 = accura1_1 + accuracy1_1
    F1_1 = F1_1 + F1_score1_1

    list_neural2 = [40, 41, 42, 43]
    accuracy2, F1_score2 = performance(train_index, test_index, 1, list_neural2, lamda2)
    accura2 = accura2 + accuracy2
    F2 = F2 + F1_score2

    list_neural3 = [15, 16, 17, 18]
    accuracy3, F1_score3 = performance(train_index, test_index, 2, list_neural3, lamda2)
    accura3 = accura3 + accuracy3
    F3 = F3 + F1_score3

    list_neural4 = [60, 61, 62, 63]
    accuracy4, F1_score4 = performance(train_index, test_index, 2, list_neural4, lamda2)
    accura4 = accura4 + accuracy4
    F4 = F4 + F1_score4

    list_neural4 = [70, 71, 72, 73]
    lamda5 = 2
    accuracy5, F1_score5 = performance(train_index, test_index, 3, list_neural4, lamda5)
    accura5 = accura5 + accuracy5
    F5 = F5 + F1_score5

    list_neural4 = [80, 81, 82, 83]
    accuracy6, F1_score6 = performance(train_index, test_index, 4, list_neural4, lamda5)
    accura6 = accura6 + accuracy6
    F6 = F6 + F1_score6

    turn += 1

print("1")
print(str(accura1 / 10), " ", str(F1 / 10))

print("2")
print(str(accura1_1 / 10), " ", str(F1_1 / 10))

print("3")
print(str(accura2 / 10), " ", str(F2 / 10))

print("4")
print(str(accura3 / 10), " ", str(F3 / 10))

print("5")
print(str(accura4 / 10), " ", str(F4 / 10))

print("6")
print(str(accura5 / 10), " ", str(F5 / 10))

print("7")
print(str(accura6 / 10), " ", str(F6 / 10))

ar = []
for i in range(0, len(data), 1):
    ar.append(i)

s1 = random.sample(ar, 150)
random1 = shuffle(s1)
training1, testing1 = train_test_split(random1, test_size=0.2, train_size=0.8)

s2 = random.sample(ar, 200)
random2 = shuffle(s2)
training2, testing2 = train_test_split(random2, test_size=0.2, train_size=0.8)

s3 = random.sample(ar, 250)
random3 = shuffle(s3)
training3, testing3 = train_test_split(random3, test_size=0.2, train_size=0.8)

s4 = random.sample(ar, 300)
random4 = shuffle(s4)
training4, testing4 = train_test_split(random4, test_size=0.2, train_size=0.8)

s5 = random.sample(ar, 350)
random5 = shuffle(s5)
training5, testing5 = train_test_split(random5, test_size=0.2, train_size=0.8)

arr1 = [150, 200, 250, 300, 350]
arr2 = [costFunction(training1), costFunction(training2), costFunction(training3), costFunction(training4),
        costFunction(training5)]

print(costFunction(training1))
print(costFunction(training2))
print(costFunction(training3))
print(costFunction(training4))
print(costFunction(training5))

plt.plot(arr1, arr2)
plt.show()
