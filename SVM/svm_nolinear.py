'''
    Group Members:
    Qian Zhu (7243641793)
    Boxuan Wang (1431189719)
    Yansong Wang (5049957463)
'''
import numpy as np
from cvxopt import matrix, solvers
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


def kernel_func(x):
    x = x ** 2
    return x


def make_product_matirx(X, Y):
    num_len = len(X)
    product_matrix = np.zeros((num_len, num_len))
    for i in range(num_len):
        for j in range(num_len):
            product_matrix[i][j] = Y[i] * Y[j] * (np.dot(kernel_func(X[i]), kernel_func(X[j])))
    # print(product_matrix.shape, product_matrix)
    return product_matrix

def predict(w, b, x):
    x = kernel_func(x)
    y_hat = np.dot(w, x) + b
    result = -1
    if y_hat >= 0:
        result = 1
    return result


if __name__ == '__main__':
    data = np.loadtxt("nonlinsep.txt", delimiter=",")
    scaler = StandardScaler()
    # data = scaler.fit_transform(data)
    X = data[:, :2]
    Y = data[:, -1]

    product_matrix = make_product_matirx(X, Y)
    P = matrix(product_matrix)
    q = matrix(-np.ones(len(X)))
    A = matrix(Y, (1, len(Y)))
    b = matrix(0.0)
    G = matrix(np.diag(-np.ones(len(X))))
    h = matrix(np.zeros(len(Y)))
    solution = solvers.qp(P, q, G, h, A, b)
    lambdas = np.array(solution['x']).reshape(-1)
    lambdas[lambdas < 1e-5] = 0

    w = np.dot(lambdas * Y,kernel_func(X))

    support_index = np.argwhere(lambdas > 0)
    b = np.mean(Y[support_index].reshape(-1) - np.dot(kernel_func(X[support_index]), w.T))

    predict_labels = [predict(w, b, x) for x in X]

    # data = scaler.inverse_transform(data)
    # X = data[:, :2]
    # Y = data[:, -1]

    plt.scatter(X[:, 0], X[:, 1], c=predict_labels)
    plt.show()
    plt.scatter(X[:, 0]**2, X[:, 1]**2, c=predict_labels)
    plt.show()

    support_vectors, suport_labels = X[support_index], Y[support_index]
    print("kernel function:f(x)=x**2")
    print("support_vectors:",support_vectors)

