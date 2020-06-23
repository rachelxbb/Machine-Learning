'''
    Group Members:
    Qian Zhu (7243641793) 
    Boxuan Wang (1431189719) 
    Yansong Wang (5049957463)
'''
import numpy as np
import matplotlib.pyplot as plt


def pca(data):
    means = np.mean(data, axis=0)
    scaled_data = data - means
    cov = np.dot(scaled_data.T, scaled_data) / len(scaled_data)
    eigs, eig_vectors = np.linalg.eig(cov)
    max_eigs_indexes = np.argsort(-eigs)[:2]
    max_eig_vectors = eig_vectors[:, max_eigs_indexes]
    print("Largest two principal components vectors：", max_eig_vectors[:, 0], max_eig_vectors[:, 1])
    low_dim_values = scaled_data.dot(max_eig_vectors)
    return low_dim_values


if __name__ == '__main__':
    data = np.loadtxt("pca-data.txt", delimiter="\t")
    low_dim_values = pca(data)
    print("PCA result：")
    print(low_dim_values[:])

