'''
    Group Members:
    Qian Zhu (7243641793) 
    Boxuan Wang (1431189719) 
    Yansong Wang (5049957463)
'''
import numpy as np

def load_data(filename):
    data1 = np.loadtxt(filename, delimiter=',')
    data = np.delete(data1, 3, 1)
    x, y = data[:,:3], data[:, -1]
    return x, y

def sigmoid(x):
    theta = np.exp(x) / (1 + np.exp(x))
    return theta

def logistic(filename):
    x, y = load_data(filename)
    col_add = np.ones(len(x))
    x_new = np.c_[col_add, x]
    learning_rate = 0.001
    row, col = np.shape(x_new)
    w = np.transpose(np.zeros((col, 1)))
    judge = True
    iteration = 1

    while iteration < 7000:
        s = np.dot(x_new, np.transpose(w))
        theta = sigmoid(s)
        result_e = np.dot(y, s)
        result = np.dot(y, x_new)

        gradient = (-1 / row) * np.sum(1 / (1 + np.exp(result_e))) * result
        w -= learning_rate * gradient
        iteration += 1

    a_list = []
    for i in range(len(y)):
        if y[i] == 1:
            a_list.append(y[i] - theta[i])
    accuracy = float(sum(a_list) / len(a_list))
    return w, accuracy

if __name__ == '__main__':
    filename = 'classification.txt'
    w, accuracy = logistic(filename)
    print('Weights: ', w)
    print('Accuracy is: ', accuracy)