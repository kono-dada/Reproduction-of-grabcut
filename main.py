from sklearn.mixture import GaussianMixture
from cv2 import cv2
import numpy as np
from networkx.classes.digraph import DiGraph
from networkx.algorithms.flow.maxflow import minimum_cut
from networkx import grid_graph
from time import time
from cvtest import grab

image: np.ndarray = cv2.imread('apple2.jpeg').astype(np.float64)
temp_image = image.copy().astype('uint8')
g_height, g_width, _ = image.shape
original_mask = np.zeros([g_height, g_width], dtype=np.uint8)
mask = np.zeros([g_height, g_width], dtype=np.uint8)
gamma = 50


def binary2image_for_debug(filename):
    _mask = np.zeros([g_height, g_width, 3])
    for h in range(g_height):
        for w in range(g_width):
            if mask[h, w] == 1:
                _mask[h, w] = [255, 255, 255]
    # cv2.imshow('1', _mask)
    # cv2.waitKey(0)
    cv2.imwrite('test\\' + filename + '.jpg', _mask)


def compute_neighbour_cost(_u, _v):
    return 0 if mask[_u] == mask[_v] else gamma * np.exp(-beta * (image[_u] - image[_v]) @ (image[_u] - image[_v]))


def compute_region_cost(_node):
    data = np.array([image[_node]])
    if original_mask[_node] == 1:
        capacity_s = np.log(np.exp(model_B._estimate_weighted_log_prob(data)[0]).sum()) + \
                     model_B._estimate_log_weights()[model_B.predict(data)[0]]
        capacity_t = np.log(np.exp(model_U._estimate_weighted_log_prob(data)[0]).sum()) + \
                     model_U._estimate_log_weights()[model_U.predict(data)[0]]
        graph.add_edge('S', _node, capacity=-capacity_s)
        graph.add_edge(_node, 'T', capacity=-capacity_t)
    else:
        graph.add_edge('S', _node, capacity=0)
        graph.add_edge(_node, 'T', capacity=2 * gamma)


def classify(model_f: GaussianMixture, model_b: GaussianMixture, h, w):
    data = np.array([image[h, w]])
    prob_f = np.log(np.exp(model_f._estimate_weighted_log_prob(data)[0]).sum())
    prob_b = np.log(np.exp(model_b._estimate_weighted_log_prob(data)[0]).sum())
    if prob_f < prob_b:
        mask[h, w] = 0
    else:
        mask[h, w] = 1


def train_gmm():
    u_array = []
    b_array = []
    for h in range(g_height):
        for w in range(g_width):
            if mask[h, w] == 1:
                u_array.append(image[h, w])
            else:
                b_array.append(image[h, w])
    _model_U = GaussianMixture(5)
    _model_U.fit(u_array)
    _model_B = GaussianMixture(5)
    _model_B.fit(b_array)
    return _model_U, _model_B


def compute_beta():
    total = 0
    for (n1, n2) in graph.edges:
        total += (image[n1] - image[n2]) @ (image[n1] - image[n2])
    total /= len(graph.edges)
    return 1 / total / 2


def draw(event, x, y, flags, *args):
    if event == cv2.EVENT_LBUTTONDOWN or flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(temp_image, [x, y], 5, [0, 0, 0], cv2.FILLED)
        cv2.circle(original_mask, [x, y], 5, 0, cv2.FILLED)


def interact():
    cv2.namedWindow('interact')
    cv2.setMouseCallback('interact', draw)
    while True:
        cv2.imshow('interact', temp_image)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):  # 如果key是q的ascii码就break
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    a, b, c, d = cv2.selectROI('select', image.astype(np.uint8), False, False)
    cv2.destroyAllWindows()
    left_up = [b, a]
    right_down = [b + d, c + a]
    cv2.rectangle(temp_image, [a, b], [a + c, b + d], [0, 0, 0], 1)
    original_mask[left_up[0]:right_down[0], left_up[1]:right_down[1]] = 1
    interact()
    cv2.imwrite('test/test.jpg', temp_image)
    grid = grid_graph(dim=(g_width, g_height))
    graph = DiGraph(grid)
    beta = compute_beta()

    grab(image.astype(np.uint8), left_up[::-1] + [right_down[1] - left_up[1], right_down[0] - left_up[0]])
    # step 1
    t0 = time()
    mask = original_mask.copy()
    # create BMM for U and B
    for ___ in range(10):
        print(f'round {___}\n')
        t0 = time()
        model_U, model_B = train_gmm()
        for _h in range(left_up[0], right_down[0]):
            for _w in range(left_up[1], right_down[1]):
                if original_mask[_h, _w] == 1:
                    classify(model_U, model_B, _h, _w)
        model_U, model_B = train_gmm()
        print(f'time2 {time() - t0}')
        t0 = time()
        binary2image_for_debug('before')

        grid = grid_graph(dim=(g_width, g_height))
        graph = DiGraph(grid)
        graph.add_node('S')
        graph.add_node('T')

        print(f'time3 {time() - t0}')
        t0 = time()
        for edge in graph.edges:
            graph.add_edge(*edge, capacity=compute_neighbour_cost(*edge))
        print(f'time4 {time() - t0}')
        t0 = time()

        for node in graph.nodes:
            if node != 'S' and node != 'T':
                compute_region_cost(node)

        print(f'time5 {time() - t0}')
        t0 = time()
        cut_value, partition = minimum_cut(graph, 'S', 'T')
        reachable, non_reachable = partition
        cut_set = set()
        for u, neighbours in ((n, graph[n]) for n in reachable):
            cut_set.update((u, v) for v in neighbours if v in non_reachable)
        for (_n1, _n2) in cut_set:
            if 'S' == _n1:
                mask[_n2] = 0
            if 'T' == _n2:
                mask[_n1] = 1
        binary2image_for_debug('after')
    for _h in range(g_height):
        for _w in range(g_width):
            if mask[_h, _w] != 1:
                image[_h, _w] = 0
    cv2.imwrite('test\\result.jpg', image)
