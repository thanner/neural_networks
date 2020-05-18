import copy
import random

B = [1, 1, 0, 0, 0, 0, 0]
S = [1, 0, 1, 0, 0, 0, 0]
P = [1, 0, 0, 1, 0, 0, 0]
T = [1, 0, 0, 0, 1, 0, 0]
V = [1, 0, 0, 0, 0, 1, 0]
X = [1, 0, 0, 0, 0, 0, 1]
E = [1, 1, 0, 0, 0, 0, 0]


def printXS(inputStringS, inputStringD):
    if (random.random() > 0.5):
        inputStringS.append('X')
        inputStringD.append(copy.deepcopy(X))
        (inputStringS, inputStringD) = printTV(inputStringS, inputStringD)
    else:
        inputStringS.append('S')
        inputStringD.append(copy.deepcopy(S))
    return (inputStringS, inputStringD)


def printT(inputStringS, inputStringD):
    inputStringS.append('T')
    inputStringD.append(copy.deepcopy(T))
    numberOfS = 0
    while ((random.random() > 0.5) and (numberOfS <= 1)):
        inputStringS.append('S')
        inputStringD.append(copy.deepcopy(S))
        numberOfS = numberOfS + 1
    inputStringS.append('X')
    inputStringD.append(copy.deepcopy(X))
    (inputStringS, inputStringD) = printXS(inputStringS, inputStringD)
    return (inputStringS, inputStringD)


def printTV(inputStringS, inputStringD):
    numberOfS = 0
    while ((random.random() > 0.5) and (numberOfS <= 1)):
        inputStringS.append('T')
        inputStringD.append(copy.deepcopy(T))
        numberOfS = numberOfS + 1
    inputStringS.append('V')
    inputStringD.append(copy.deepcopy(V))
    if (random.random() > 0.5):
        inputStringS.append('P')
        inputStringD.append(copy.deepcopy(P))
        (inputStringS, inputStringD) = printXS(inputStringS, inputStringD)
    else:
        inputStringS.append('V')
        inputStringD.append(copy.deepcopy(V))
    return (inputStringS, inputStringD)


def printP(inputStringS, inputStringD):
    inputStringS.append('P')
    inputStringD.append(copy.deepcopy(P))
    (inputStringS, inputStringD) = printTV(inputStringS, inputStringD)
    return (inputStringS, inputStringD)


def generate_string():
    inputStringS = []
    inputStringD = []
    inputStringS.append('B')
    inputStringD.append(copy.deepcopy(B))
    if (random.random() > 0.5):
        (inputStringS, inputStringD) = printT(inputStringS, inputStringD)
    else:
        (inputStringS, inputStringD) = printP(inputStringS, inputStringD)
    outputStringS = copy.deepcopy(inputStringS)
    outputStringD = copy.deepcopy(inputStringD)
    outputStringS.pop(0)
    outputStringS.append('E')
    outputStringD.pop(0)
    outputStringD.append(copy.deepcopy(E))
    for j in range(len(outputStringD)):
        outputStringD[j].pop(0)
    return (inputStringD, outputStringD, inputStringS, outputStringS)

# arquivo1 = open('inputD_test.txt', 'w')
# arquivo2 = open('outputD_test.txt', 'w')

# arquivo1.write(str(inputD))
# arquivo2.write(str(outputD))

# arquivo1.close()
# arquivo2.close()

# print('done!')
