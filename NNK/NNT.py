'''
    Group Members:
    Qian Zhu (7243641793) 
    Boxuan Wang (1431189719) 
    Yansong Wang (5049957463)
'''
from PIL import Image
import numpy as np


def read_pgm(path):
    im = np.array(Image.open(path))
    im = im / 255
    return im


def compute_acc(predictions, labels):
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            count += 1
    return count / len(labels)


class nn:
    def __init__(self, input_shape, num_hidden, lr):
        self.weights1 = [np.random.uniform(low=-0.01, high=0.01, size=input_shape) for i in range(num_hidden)]
        self.bias1 = [1 for i in range(num_hidden)]
        self.weights2 = np.random.uniform(low=-0.01, high=0.01, size=num_hidden)
        self.bias2 = 1
        self.lr = lr

    def compute_output(self, input_array):
        self.input_array = input_array
        self.hiddens = [np.sum(weight * input_array) for weight in self.weights1]
        self.hiddens = [self.hiddens[i] + self.bias1[i] for i in range(len(self.hiddens))]
        self.hiddens_sigmoid = [self.sigmoid(hidden) for hidden in self.hiddens]

        self.output = np.sum(np.array(self.hiddens_sigmoid) * self.weights2) + self.bias2
        self.output_sigmoid = self.sigmoid(self.output)
        predict_logit = self.output_sigmoid
        return predict_logit

    def update_loss(self, label, predict_logit):
        loss_on_y = 2 * (predict_logit - label)
        loss_on_output = loss_on_y * self.sigmoid_derivative(self.output_sigmoid)
        loss_on_w2 = np.array([loss_on_output * hi for hi in self.hiddens_sigmoid])
        loss_on_w1 = [
            loss_on_output * self.weights2[i] * self.sigmoid_derivative(self.hiddens_sigmoid[i]) * self.input_array for
            i in range(len(self.weights2))]

        loss_on_bias2 = loss_on_output
        loss_on_bias1 = [loss_on_output * self.weights2[i] * self.sigmoid_derivative(self.hiddens_sigmoid[i]) for i in
                         range(len(self.weights2))]
        self.weights2 -= loss_on_w2 * self.lr
        self.bias2 -= loss_on_bias2 * self.lr

        for i in range(len(self.weights1)):
            self.weights1 -= loss_on_w1[i] * self.lr
            self.bias1 -= loss_on_bias1[i] * self.lr

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def sigmoid_derivative(self, sigmoid_output):
        return sigmoid_output * (1 - sigmoid_output)


if __name__ == '__main__':
    imgs, labels = [], []
    for path in open("downgesture_train.list.txt"):
        path = path.strip()
        img = read_pgm(path)
        label = int("down" in path)
        imgs.append(img)
        labels.append(label)

    net = nn(imgs[0].shape, 100, 0.1)

    for epoch in range(1001):
        total_loss = 0
        predictions = []
        for img, label in zip(imgs, labels):
            predict_logit = net.compute_output(img)
            net.update_loss(label, predict_logit)
            total_loss += (predict_logit - label) ** 2
            predictions.append(predict_logit > 0.5)
        print("epoch:%d,loss:%.3f" % (epoch, total_loss), compute_acc(predictions, labels))

    test_imgs, test_labels = [], []
    paths = []
    for path in open("downgesture_test.list.txt"):
        path = path.strip()
        img = read_pgm(path)
        label = int("down" in path)
        test_imgs.append(img)
        test_labels.append(label)
        paths.append(path)

    predict_logits = []
    for img, label in zip(test_imgs, test_labels):
        logit = net.compute_output(img)
        predict_logits.append(logit)

    predictions = [int(logit > 0.5) for logit in predict_logits]
    print("prediction:", predictions)
    with open("prediction.txt", "w") as f:
        for path, predict in zip(paths, predictions):
            f.write("%s:%s\n" % (path, predict))

    print("test acc:", compute_acc(predictions, test_labels))
