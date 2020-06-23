'''
    Group Members:
    Qian Zhu (7243641793) 
    Boxuan Wang (1431189719) 
    Yansong Wang (5049957463)
'''
import numpy as np

def load_data(filename):
    data1 = np.loadtxt(filename, delimiter=',')
    data = np.delete(data1, 4, 1)
    x, y = data[:, :-1], data[:, -1]
    return x, y

def perceptron(filename):
    x, y = load_data(filename)
    col_add = np.ones(len(x))
    x_new = np.c_[col_add, x]
    learning_rate = 0.001
    row, col = np.shape(x_new)
    w = np.transpose(np.zeros((col, 1)))

    alpha = 0.01
    judge = True
    iteration = 0
    while judge == True:
        violated = []
        iteration += 1
        for i in range(len(x_new)):
            result = np.dot(w, np.transpose(x_new[i]))
            if result < 0 and y[i] == 1:
                w = w + alpha * x_new[i]
                violated.append(i)
            elif result >= 0 and y[i] == -1:
                w = w - alpha * x_new[i]
                violated.append(i)
                
        accuracy = 1 - len(violated) / len(x)
        if len(violated) == 0:
            judge = False

    return w, accuracy

if __name__ == '__main__':
    filename = 'classification.txt'
    x, y = load_data(filename)
    w, accuracy = perceptron(filename)
    print('Weights: ', w)
    print('Accuracy is: ', accuracy)