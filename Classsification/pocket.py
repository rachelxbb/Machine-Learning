'''
    Group Members:
    Qian Zhu (7243641793) 
    Boxuan Wang (1431189719) 
    Yansong Wang (5049957463)
'''
import numpy as np
from random import choice
import matplotlib.pyplot as plt

def load_data(filename):
    data1 = np.loadtxt(filename, delimiter=',')
    data = np.delete(data1, 3, 1)
    x, y = data[:, :-1], data[:, -1]
    return x, y

def pocket(filename):
    x, y = load_data(filename)
    col_add = np.ones(len(x))
    x_new = np.c_[col_add, x]
    learning_rate = 0.001
    row, col = np.shape(x_new)
    w = np.transpose(np.zeros((col, 1)))
    mis_classification = []
    
    alpha = 0.01
    iteration = 0
    judge = True
    length = len(x_new)
    while judge and iteration < 7000:
        iteration += 1
        violated = []
        for i in range(len(x_new)):
            result = np.dot(x_new[i], np.transpose(w))
            if result < 0 and y[i] == 1:
                violated.append(i)                    
            elif result >= 0 and y[i] == -1:
                violated.append(i)

        if len(violated) != 0:
            if len(violated) < length:
                best_w = w
                length = len(violated)
            mis_classification.append(len(violated))
            j = choice(violated)
            if np.dot(x_new[j], np.transpose(w)) < 0 and y[j] == 1:
                w = w + alpha * x_new[j]
            elif np.dot(x_new[j], np.transpose(w)) >= 0 and y[j] == -1:
                w = w - alpha * x_new[j]
        else:
            judge = False
    plt.plot(mis_classification)
    plt.xlabel('Number of Iterations') 
    plt.ylabel('Number of Misclassification')
    plt.title('Misclassification VS. Iterations') 
    plt.show()
    accuracy = 1 - len(violated) / len(x_new)
    return best_w, accuracy

if __name__ == '__main__':
    filename = 'classification.txt'
    w, accuracy = pocket(filename)
    print('Best Weights: ', w)
    print('Accuracy is: ', accuracy)