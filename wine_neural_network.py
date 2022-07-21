import random

import matplotlib.pyplot as plt
import numpy
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data = genfromtxt('datasets/hw3_wine.csv', skip_header=1)


def check(test, weight):
    mt = []
    res = []

    arr1 = []
    arr1.append(1)
    mt.append(arr1)

    for i in range(0, 13, 1):
        arr = []
        arr.append(data[test][i + 1])
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

    if res[0][0] >= res[1][0] and res[0][0] >= res[2][0]:
        return 1
    if res[1][0] >= res[0][0] and res[1][0] >= res[2][0]:
        return 2
    return 3


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
                for k in range(0, 14, 1):
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
    for i in range(0, 3, 1):
        arr = []
        for k in range(0, last, 1):
            arr.append(random.uniform(-1, 1))
        mat1.append(arr)

    weight.append(mat1)

    final = []
    for i in range(0, 3, 1):
        arr = []
        for j in range(0, len(train), 1):
            if data[train[j]][0] == i + 1:
                arr.append(1)
            else:
                arr.append(0)

        final.append(arr)

    for cnt in range(0, 1000, 1):
        # print("cnt-----")
        # print(cnt)
        res = []
        a = []

        for i in range(0, 14, 1):
            arr = []

            for j in range(0, len(train), 1):
                if i == 0:
                    arr.append(1)
                else:
                    arr.append(data[train[j]][i])
            res.append(arr)

        a.append(res)

        # print(a)
        for i in range(0, len(weight), 1):

            result = np.matmul(weight[i], a[i])
            if i < len(weight) - 1:
                arr1 = []
                for j in range(0, len(res[0]), 1):
                    arr1.append(1)

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
    # print(test)
    # exit()

    weight = trainning(train, layer, list_neural, lamda)

    true_pos1, true_pos2, true_pos3 = 0, 0, 0
    false_pos1, false_pos2, false_pos3 = 0, 0, 0
    pos1, pos2, pos3 = 0, 0, 0

    for i in range(0, len(test), 1):
        result = check(test[i], weight)

        if data[test[i]][0] == 1:
            pos1 += 1
            if result == 1:
                true_pos1 += 1
            elif result == 2:
                false_pos2 += 1
            elif result == 3:
                false_pos3 += 1

        elif data[test[i]][0] == 2:
            pos2 += 1
            if result == 2:
                true_pos2 += 1
            elif result == 1:
                false_pos1 += 1
            elif result == 3:
                false_pos3 += 1

        elif data[test[i]][0] == 3:
            pos3 += 1
            if result == 2:
                false_pos2 += 1
            elif result == 1:
                false_pos1 += 1
            elif result == 3:
                true_pos3 += 1

    accura = (true_pos1 + true_pos2 + true_pos3) / len(test)
    # print(accura)

    precision = 0
    if true_pos1 + false_pos1 > 0:
        precision += true_pos1 / (true_pos1 + false_pos1)

    if true_pos2 + false_pos2 > 0:
        precision += true_pos2 / (true_pos2 + false_pos2)

    if true_pos3 + false_pos3 > 0:
        precision += true_pos3 / (true_pos3 + false_pos3)

    precision = precision / 3

    recall = (true_pos1 / pos1 + true_pos2 / pos2 + true_pos3 / pos3) / 3

    F1 = 2 * (precision * recall) / (precision + recall)
    return accura, F1, weight


def costFunction(train):
    # print("len-----")
    # print(len(weight))
    # print(weight)
    weight = trainning(train, 2, [60, 61, 62, 63], 1)

    final = []
    for i in range(0, 3, 1):
        arr = []
        for j in range(0, len(train), 1):
            if data[train[j]][0] == i + 1:
                arr.append(1)
            else:
                arr.append(0)

        final.append(arr)

    result = []
    for i in range(0, 14, 1):
        arr = []

        for j in range(0, len(train), 1):
            if i == 0:
                arr.append(1)
            else:
                arr.append(data[train[j]][i])
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
        for j in range(0, 3, 1):
            if final[j][i] == 0:
                J += -np.log(1 - result[j][i])
            else:
                J += -np.log(result[j][i])

        cost = cost + J
    cost = cost / len(train)

    sum = 0
    for i in range(0, len(weight) - 1, 1):
        for j in range(0, len(weight[i]), 1):
            for k in range(0, len(weight[i][0]), 1):
                if k != 0:
                    sum = sum + weight[i][j][k] * weight[i][j][k]

    lamda = 1
    sum = lamda / (2 * len(train)) * sum
    return cost + sum


y1 = []
y2 = []
y3 = []
for i in range(0, len(data), 1):
    if data[i][0] == 1:
        y1.append(i)
    elif data[i][0] == 2:
        y2.append(i)
    elif data[i][0] == 3:
        y3.append(i)

ls = []
st = 0

ll1 = len(y1)
ll2 = len(y2)
ll3 = len(y3)
# print(ll3)

while st < 10:
    ls1 = []
    if st == 9:
        for i in range(0, len(y1), 1):
            ls1.append(y1[i])
        for i in range(0, len(y2), 1):
            ls1.append(y2[i])
        for i in range(0, len(y3), 1):
            ls1.append(y3[i])

    else:
        gr1 = np.random.choice(y1, 6, replace=False)
        gr2 = np.random.choice(y2, int(ll2 / 10), replace=False)
        gr3 = np.random.choice(y3, 5, replace=False)

        for i in range(0, len(gr1), 1):
            ls1.append(gr1[i])
            y1.remove(gr1[i])
        for i in range(0, len(gr2), 1):
            ls1.append(gr2[i])
            y2.remove(gr2[i])
        for i in range(0, len(gr3), 1):
            ls1.append(gr3[i])
            y3.remove(gr3[i])

    ls.append(ls1)
    st += 1

accura1, F1 = 0.0, 0.0
accura1_1, F1_1 = 0.0, 0.0
accura2, F2 = 0.0, 0.0
accura3, F3 = 0.0, 0.0
accura4, F4 = 0.0, 0.0
accura5, F5 = 0.0, 0.0
accura6, F6 = 0.0, 0.0

turn = 0
perfect_weight = []
for i in range(0, len(ls), 1):

    train_index = []
    for j in range(0, len(ls), 1):
        if j != i:
            for k in range(0, len(ls[j]), 1):
                train_index.append(ls[j][k])

    list_neural1 = [20, 21, 22, 23]
    lamda1 = 0.2
    accuracy1, F1_score1, w1 = performance(train_index, ls[i], 1, list_neural1, lamda1)
    accura1 = accura1 + accuracy1
    F1 = F1 + F1_score1

    lamda2 = 1
    accuracy1_1, F1_score1_1, w1_1 = performance(train_index, ls[i], 1, list_neural1, lamda2)
    accura1_1 = accura1_1 + accuracy1_1
    F1_1 = F1_1 + F1_score1_1

    list_neural2 = [40, 41, 42, 43]
    accuracy2, F1_score2, w2 = performance(train_index, ls[i], 1, list_neural2, lamda2)
    accura2 = accura2 + accuracy2
    F2 = F2 + F1_score2

    list_neural3 = [15, 16, 17, 18]
    accuracy3, F1_score3, w3 = performance(train_index, ls[i], 2, list_neural3, lamda2)
    accura3 = accura3 + accuracy3
    F3 = F3 + F1_score3

    list_neural4 = [60, 61, 62, 63]
    accuracy4, F1_score4, w4 = performance(train_index, ls[i], 2, list_neural4, lamda2)
    perfect_weight = w4
    accura4 = accura4 + accuracy4
    F4 = F4 + F1_score4

    list_neural4 = [70, 71, 72, 73]
    lamda5 = 2
    accuracy5, F1_score5, w5 = performance(train_index, ls[i], 3, list_neural4, lamda5)
    accura5 = accura5 + accuracy5
    F5 = F5 + F1_score5

    list_neural4 = [80, 81, 82, 83]
    accuracy6, F1_score6, w6 = performance(train_index, ls[i], 4, list_neural4, lamda5)
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

s1 = random.sample(ar, 20)
random1 = shuffle(s1)
training1, testing1 = train_test_split(random1, test_size=0.2, train_size=0.8)

s2 = random.sample(ar, 50)
random2 = shuffle(s2)
training2, testing2 = train_test_split(random2, test_size=0.2, train_size=0.8)

s3 = random.sample(ar, 80)
random3 = shuffle(s3)
training3, testing3 = train_test_split(random3, test_size=0.2, train_size=0.8)

s4 = random.sample(ar, 110)
random4 = shuffle(s4)
training4, testing4 = train_test_split(random4, test_size=0.2, train_size=0.8)

s5 = random.sample(ar, 140)
random5 = shuffle(s5)
training5, testing5 = train_test_split(random5, test_size=0.2, train_size=0.8)

arr1 = [20, 50, 80, 110, 140]
arr2 = [costFunction(training1), costFunction(training2), costFunction(training3), costFunction(training4),
        costFunction(training5)]

print(costFunction(training1))
print(costFunction(training2))
print(costFunction(training3))
print(costFunction(training4))
print(costFunction(training5))

plt.plot(arr1, arr2)
plt.show()
