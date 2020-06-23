'''
    Group Members:
    Qian Zhu (7243641793) 
    Boxuan Wang (1431189719) 
    Yansong Wang (5049957463)
'''
import numpy as np

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    D, y= data[:,:2], data[:, -1]
    return D, y

def linear(filename):
    D, y = load_data(filename)
    y_vec = np.transpose(y)
    col_add = np.ones(len(D))
    D_new = np.c_[col_add, D]
    D_t = np.transpose(D_new)
    result = np.dot(D_t, D_new)
    w = np.dot(np.dot(np.linalg.inv(result), D_t), y_vec)
    return w

if __name__ == '__main__':
    filename = 'linear-regression.txt'
    w = linear(filename)
    print('Weights: ', w)