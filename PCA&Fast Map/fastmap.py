'''
    Group Members:
    Qian Zhu (7243641793) 
    Boxuan Wang (1431189719) 
    Yansong Wang (5049957463)
'''
import numpy as np
import matplotlib.pyplot as plt


def load_distances(path="fastmap-data.txt"):
    data = np.loadtxt(path, delimiter="\t")
    num = len(set(data[:, 0].tolist() + data[:, 1].tolist()))
    distances = np.zeros((num, num))
    for start, end, dis in data:
        start, end = int(start) - 1, int(end) - 1
        distances[start, end] = dis
        distances[end, start] = dis
    return distances


def load_words(path="fastmap-wordlist.txt"):
    with open(path, "r") as f:
        lines = f.readlines()
    words = [l.strip() for l in lines]
    return words


def fastmap(distances, target_dim):
    num = len(distances)
    rs = np.zeros((num, target_dim))
    for k in range(target_dim):
        pos = np.where(distances == np.max(distances))
        start = pos[0][0]
        end = pos[1][0]
        print("Max Value：", (start,end), distances[start, end] )
        for i in range(num):
            rs[i, k] = (distances[start, i] ** 2 + distances[start, end] ** 2 - distances[i, end] ** 2) / (
                    2 * distances[start, end])
        new_distances = np.zeros(distances.shape)
        for i in range(num):
            for j in range(num):
                new_distances[i, j] = np.sqrt(distances[i, j] ** 2 - (rs[i, k] - rs[j, k]) ** 2)
        distances = new_distances
    return rs


if __name__ == '__main__':
    distances = load_distances(path="fastmap-data.txt")
    rs = fastmap(distances, 2)

    words = load_words(path="fastmap-wordlist.txt")
    x = rs[:, 0]
    y = rs[:, 1]
    plt.scatter(x, y)
    for i in range(len(x)):
        plt.annotate(words[i], xy=(x[i], y[i]), xytext=(x[i] + 0.2, y[i]-0.2))
    plt.show()

