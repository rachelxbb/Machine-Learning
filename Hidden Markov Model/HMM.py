'''
    Group Members:
    Qian Zhu (7243641793)
    Boxuan Wang (1431189719)
    Yansong Wang (5049957463)
'''
import numpy as np

def load_data():
    lines = open('hmm-data.txt', 'r').readlines()
    grid = np.array([[float(num) for num in l.strip().split()] for l in lines[2:12]])

    towers = [l.strip().split(":")[1] for l in lines[16:20]]
    towers = np.array([[float(num) for num in pos.split()] for pos in towers])

    distances_list = np.array([[float(num) for num in l.strip().split()] for l in lines[24:35]])
    return grid, towers, distances_list


def index2point(index, row_length=10):
    row = index // row_length
    col = index % row_length
    return (row, col)


def point2index(point, row_length=10):
    row, col = point
    return row * row_length + col


def make_transition_matrix(points, point2id, id2point, grid):
    directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    point_num = len(points)
    transition = np.zeros((point_num, point_num))
    for i in range(point_num):
        point = id2point[i]
        current_index = point2id[point]
        for direction in directions:
            neighbour = tuple((np.array(point) + direction).tolist())
            if neighbour in points and grid[neighbour[0], neighbour[1]] != 0:
                neighbour_index = point2id[neighbour]
                transition[current_index, neighbour_index] = 1
    for i in range(point_num):
        transition[i, :] /= np.sum(transition[i, :])
    return transition


def make_pai(points, point2id, grid):
    proba = np.ones(len(points))
    for point in points:
        id = point2id[point]
        if grid[point[0], point[1]] == 0:
            proba[id] = 0
    proba /= np.sum(proba)
    return proba


def make_B(distances_list, towers, points, point2id):
    B = np.ones((len(points), len(distances_list)))
    for j, distances in enumerate(distances_list):
        for point in points:
            for i, tower in enumerate(towers):
                dis = np.sqrt(np.sum((tower - np.array(point)) ** 2))
                noise_dis = [0.7 * dis, 1.3 * dis]
                if distances[i] >= noise_dis[0] and distances[i] <= noise_dis[1]:
                    B[point2id[point], j] *= 1 / (noise_dis[1] - noise_dis[0])
                else:
                    B[point2id[point], j] *= 0
    return B


def viterbi(pai, transition, B, points):
    total_step = B.shape[1]

    alpha = np.zeros((total_step, len(points)))
    log = np.zeros((total_step, len(points)))
    path = np.zeros(total_step).astype(int)
    for i in range(len(points)):
        alpha[0, i] = pai[i] * B[i, 0]
        log[0, i] = 0

    for step in range(total_step - 1):
        for i in range(len(points)):
            logits = []
            transition_logits = []
            for j in range(len(points)):
                logits.append(B[i, step + 1] * alpha[step, j] * transition[j, i])
                transition_logits.append(alpha[step, j] * transition[j, i])
            alpha[step + 1, i] = max(logits)
            log[step + 1, i] = np.array(transition_logits).argmax()

    path[-1] = int(alpha[total_step - 1, :].argmax())
    for step in range(total_step - 2, -1, -1):
        path[step] = log[step + 1, path[step + 1]]
    return path


if __name__ == '__main__':
    grid, towers, distances_list = load_data()
    points = []
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            point = (i, j)
            points.append(point)
    point2id = {point: i for i, point in enumerate(points)}
    id2point = {v: k for k, v in point2id.items()}

    # for index in point_indexes:
    #     point = index2point(index)
    #     print(point, point2index(point))

    pai = make_pai(points, point2id, grid)
    transition = make_transition_matrix(points, point2id, id2point, grid)
    B = make_B(distances_list, towers, points, point2id)

    path = viterbi(pai, transition, B, points)
    print([id2point[id] for id in path])

