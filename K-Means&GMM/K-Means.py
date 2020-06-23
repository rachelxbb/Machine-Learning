import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from random import sample
import math


#load data
def load_data(file):
    fhand = open(file)
    data= []
    for line in fhand.readlines():
        lines = line.strip().split(',')
        data.append([float(lines[0]), float(lines[1])])
    return data


#choose random centriod
def choose_centroid(data, k):
    centroids = random.sample(data, k)
    return centroids


#calculate closest distance
def cal_distance(p1, p2):
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


#assign to closest centroid
def assign_centroid(data, centroids):
    centroid_dict = {}
    for item in data:
        distance_list = []
        for centroid in centroids:
            distance = cal_distance(item, centroid)
            distance_list.append(distance)
        small_distance = min(distance_list)
        index = distance_list.index(small_distance)
        closest_centroid = str(centroids[index])
        if closest_centroid not in centroid_dict.keys():
            centroid_dict[closest_centroid] = [item]
        else:
            centroid_dict[closest_centroid].append(item)
    return centroid_dict


#recalculate centroid
def recal_centroid(centroid_dict):
    centroid_list = []
    for key in centroid_dict.keys():
        centroid = np.mean(centroid_dict[key], axis = 0)
        centroid_list.append(centroid)
    return np.array(centroid_list).tolist()


#visualization plots
def vis_k(data, centroids):
    centroid_dict = assign_centroid(data, centroids)
    for centroid in centroid_dict.keys():
        index = list(centroid_dict.keys()).index(centroid)
        if index == 0:
            for i in range(len(centroid_dict[centroid])):
                plt.scatter(centroid_dict[centroid][i][0], centroid_dict[centroid][i][1], color = 'red')
        elif index == 1:
            for i in range(len(centroid_dict[centroid])):
                plt.scatter(centroid_dict[centroid][i][0], centroid_dict[centroid][i][1], color = 'green')
        elif index == 2:
            for i in range(len(centroid_dict[centroid])):
                plt.scatter(centroid_dict[centroid][i][0], centroid_dict[centroid][i][1], color = 'yellow')
    plt.show()


#calculate convergence
def cal_conver(centroids_new, centroid_dict):
    sum_dis = 0
    for centroid_n in centroids_new:
        dis = 0
        centroid_dict = assign_centroid(data, centroids_new)
        for point in centroid_dict[str(centroid_n)]:
            dis += cal_distance(centroid_n, point)
        sum_dis += dis
    return sum_dis


def k_means():
    #load data
    # data = load_data("clusters.txt")
    # k = 3
    
    #initiate centroids
    centroids_old = choose_centroid(data, k)
    
    #assign to closest centroids and get the new centroids
    centroid_dict = assign_centroid(data, centroids_old)
    centroids_new = recal_centroid(centroid_dict)
    
    
    #repeat previous steps and stop till convergence
    sum_dis = cal_conver(centroids_new, centroid_dict)
    old_dis = 1
    
    while abs(sum_dis - old_dis) >= 0.000001:
        centroids_new = recal_centroid(centroid_dict)
        centroids_dict = assign_centroid(data, centroids_new)
        k_means_plot = vis_k(data, centroids_new)
        old_dis = sum_dis
        sum_dis = cal_conver(centroids_new, centroid_dict)
        print('The centroids are:', centroids_new)
    return k_means_plot


if __name__ == '__main__':
    data = load_data("clusters.txt")
    k = 3
    k_means()