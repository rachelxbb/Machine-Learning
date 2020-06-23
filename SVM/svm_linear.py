'''
    Group Members:
    Qian Zhu (7243641793)
    Boxuan Wang (1431189719)
    Yansong Wang (5049957463)
'''
import numpy as np
import cvxopt
import cvxopt.solvers

def load_data(file):
    X = []
    Y = []
    with open(file) as f:
        for line in f:
            a = line.strip('\n').split(',')
            X.append([float(a[0]), float(a[1])])
            Y.append(float(a[-1]))
        X = np.mat(X)
    return X, Y

def QPP(X, Y):
    size = X.shape[0]
    Q = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            Q[i, j] = Y[i] * Y[j] * np.dot(X[i], np.transpose(X[j]))
    P = cvxopt.matrix(Q)
    q = cvxopt.matrix(np.ones(size) * -1)
    b = cvxopt.matrix(float(0))
    A = cvxopt.matrix(Y, (1, size))
    G = cvxopt.matrix(np.diag(np.ones(size) * -1))
    h = cvxopt.matrix(np.zeros(size))
    result = cvxopt.solvers.qp(P, q, G, h, A, b)
    return result

def SVM(result, Y):
    alpha = result['x']
    alpha_list = []
    index = []
    svm_list = []
    for a in alpha:
        alpha_list.append(a)
    for i in range(len(alpha_list)):
        if alpha_list[i] > 1e-5:
            index.append(i)
            for a in X[i]:
                svm_list.extend(a.tolist())
    Y = np.array(Y)
    weights = [0, 0]
    for j in index:
        weights += alpha_list[j] * Y[j] *X[j]
        b = 1 / Y[j] - weights * X[j].T
    print('Equation: ', np.array(weights)[0].T, '*X', float(b), '= 0')
    print('SVM vectors are: ', svm_list)

if __name__ == '__main__':
	file = 'linsep.txt'
	X, Y = load_data(file)
	result = QPP(X, Y)
	SVM(result, Y)