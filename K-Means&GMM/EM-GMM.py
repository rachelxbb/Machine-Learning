import numpy as np
import copy
import matplotlib.pyplot as plt
import random


class GMM:
    def __init__(self, data_path, K=3):
        self.data = self.load_data(data_path)
        # self.data = scale(self.data)
        self.data_num = len(self.data)
        self.data_dim = self.data.shape[1]
        self.models_num = K

        # initialize the parameters_

        self.alphas = [1 / self.models_num for _ in range(self.models_num)] 

        self.us = [self.choice_one_point() for _ in range(self.models_num)]  
        self.covs = [(self.data - self.us[i]).T.dot(self.data - self.us[i]) * 1 / self.models_num for i in
                     range(self.models_num)]  

        self.data_source_probs = self.compute_data_source_prob()  

    def load_data(self, path, delimiter=","):
        data = np.loadtxt(path, delimiter=delimiter)
        print("data loaded, shape:", data.shape)
        return data

    def choice_one_point(self):
        index = int(self.data_num * random.random())
        return self.data[index]

    def gaussian_model_prob(self, data_point, model_index):  
        u = self.us[model_index]  
        cov = self.covs[model_index]
        cov_determinant = np.linalg.det(cov)  
        cov_inv = np.linalg.inv(cov)  
        exponent = - (data_point - u).T.dot(cov_inv).dot(data_point - u) / 2
        # print(cov_determinant)
        denominator = np.sqrt(np.power(2 * np.pi, self.data_dim) * np.abs(cov_determinant))
        prob = (1 / denominator) * np.exp(exponent)
        if prob == 0:
            prob += prob + 0.00001
        return prob

    def compute_point_source_prob(self, data_point):  
        point_probs = [self.gaussian_model_prob(data_point, i) for i in range(self.models_num)]
        weighted_probs = [alpha * prob for alpha, prob in zip(self.alphas, point_probs)]
        denominator = np.sum(weighted_probs)
        # if denominator <=0:
        #     print(denominator,point_probs)
        point_source_probs = [p / denominator for p in weighted_probs]
        return point_source_probs

    def compute_data_source_prob(self):  
        probs = [self.compute_point_source_prob(data_point) for data_point in self.data]
        data_source_probs = np.array(probs)
        return data_source_probs  

    def compute_means_for_single_model(self, model_index):  
        points_probs = self.data_source_probs[:, model_index]
        points_sum = np.zeros(self.data[0].shape)
        for prob, data_point in zip(points_probs, self.data):
            points_sum = points_sum + prob * data_point
        u = points_sum / np.sum(points_probs)
        print(u)
        return u


    def compute_cov_matrix_for_single_model(self, model_index):  
        points_probs = self.data_source_probs[:, model_index] 
        u = self.us[model_index]
        cov = np.zeros((self.data_dim, self.data_dim))
        for prob, data_point in zip(points_probs, self.data):
            delta = (data_point - u).reshape(self.data_dim, 1)
            cov += prob * (delta).dot(delta.T)
        cov = cov / np.sum(points_probs)
        print(cov)
        return cov


    def compute_alphas_for_single_model(self, model_index):  
        points_probs = self.data_source_probs[:, model_index]  
        alpha = np.sum(points_probs) / self.data_num
        if alpha > 1:
            print(np.sum(points_probs), self.data_num)
            print(np.sum(self.data_source_probs[0, :]))
        return alpha

    def expectation(self):  
        self.data_source_probs = self.compute_data_source_prob()  

    def maximization(self):  
        self.us = [self.compute_means_for_single_model(i) for i in range(self.models_num)]
        self.covs = [self.compute_cov_matrix_for_single_model(i) for i in range(self.models_num)]
        self.alphas = [self.compute_alphas_for_single_model(i) for i in range(self.models_num)]

    def compute_diff(self, old_alphas, old_covs, old_us):  
        total_distance = 0
        for i in range(self.models_num):
            d1 = np.linalg.norm(old_alphas[i] - self.alphas[i])
            d2 = np.linalg.norm(old_covs[i] - self.covs[i])
            d3 = np.linalg.norm(old_us[i] - self.us[i])
            total_distance += d3 + d2 + d1
        print(total_distance)
        return total_distance

    def fit(self, min_diff=1e-3, max_step=10000):  
        step_diff = 1
        step = 0
        old_alphas, old_covs, old_us = copy.deepcopy(self.alphas), copy.deepcopy(self.covs), copy.deepcopy(self.us)

        while step_diff > min_diff and step < max_step:
            self.expectation()
            self.maximization()
            step_diff = self.compute_diff(old_alphas, old_covs, old_us)
            old_alphas, old_covs, old_us = copy.deepcopy(self.alphas), copy.deepcopy(self.covs), copy.deepcopy(self.us)
            step += 1
            print("step:%d ,model step diff:%.6f" % (step, step_diff))
            # print(old_us,self.us)
        print("after %d step, model iteration ended with step difference %f!" % (step, step_diff))

    def predict(self):
        labels = []
        for data_point in self.data:
            probs = self.compute_point_source_prob(data_point)
            probs = [alpha * p for alpha, p in zip(self.alphas, probs)]
            cluster = np.argmax(probs)
            labels.append(cluster)
        return self.data, labels


if __name__ == '__main__':
    gmm_model = GMM(data_path="clusters.txt", K=3)
    gmm_model.fit()
    data, labels = gmm_model.predict()
    x, y = data[:, 0], data[:, 1]
    plt.scatter(x, y, c=labels)
    plt.title("GMM")
    plt.show()

    from collections import Counter
    print(Counter(labels))


