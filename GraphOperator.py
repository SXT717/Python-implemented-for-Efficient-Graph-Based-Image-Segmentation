import math
import numpy as np

class OptimizedUnionFind:
    def __init__(self, num_node):
        self.parent = [i for i in range(num_node)]
        self.size = [1 for _ in range(num_node)]
        self.num_set = num_node

    def size_of(self, u):
        return self.size[u]

    def find(self, u):#
        #直到子节点和父节点是同一个节点才返回
        if self.parent[u] == u: return u
        self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def merge(self, u, v):
        u = self.find(u)#寻找根节点
        v = self.find(v)#寻找根节点

        if u != v:#如果两个父节点不是相同的
            #谁的子节点多，谁当父节点，这个判断过程不是必须的
            if self.size[u] > self.size[v]:
                self.parent[v] = u #让v的父节点是u
                self.size[u] += self.size[v]#更新过程，size特应该更新
                self.size[v] = 1#被合并的根节点的子节点数目变为1
            else:
                self.parent[u] = v #让u的父节点是v
                self.size[v] += self.size[u]#更新过程，size特应该更新
                self.size[u] = 1#被合并的根节点的子节点数目变为1
            self.num_set -= 1#类别减1

def get_diff(img, x1, y1, x2, y2):#计算两个像素之间的RGB距离
    r = (img[0][y1, x1] - img[0][y2, x2]) ** 2
    g = (img[1][y1, x1] - img[1][y2, x2]) ** 2
    b = (img[2][y1, x1] - img[2][y2, x2]) ** 2
    return math.sqrt(r + g + b)

def get_threshold(k, size):
    return (k / size)

def create_edge(img, width, x1, y1, x2, y2):
    vertex_id = lambda x, y: y * width + x

    w = get_diff(img, x1, y1, x2, y2)
    return (vertex_id(x1, y1), vertex_id(x2, y2), w)

def build_graph(img, width, height):
    graph = []

    for y in range(height):
        for x in range(width):
            if x < width - 1:
                graph.append(create_edge(img, width, x, y, x + 1, y))
            if y < height - 1:
                graph.append(create_edge(img, width, x, y, x, y + 1))
            if x < width - 1 and y < height - 1:
                graph.append(create_edge(img, width, x, y, x + 1, y + 1))
            if x < width - 1 and y > 0:
                graph.append(create_edge(img, width, x, y, x + 1, y - 1))

    return graph

def segment_graph(sorted_graph, num_node, k):
    ufset = OptimizedUnionFind(num_node)
    threshold = [get_threshold(k, 1)] * num_node  # 在这里每个树只包含一个元素
    for edge in sorted_graph:#因为是从小到大遍历，所以一开始的自然是最小的类间间距
        u = ufset.find(edge[0])#寻找父节点
        v = ufset.find(edge[1])#寻找父节点
        w = edge[2]#得
        # w是两个类间不相似度，threshold是类内不相似度
        if u != v:#如果两个节点的父节点不相同，也就是不属于同一类
            if w <= threshold[u] and w <= threshold[v]:#如果边的权重小于阈值
                ufset.merge(u, v)  # 合并两个节点
                parent = ufset.find(u)
                #这里更新最大的类内间距
                threshold[parent] = np.max([w, threshold[u], threshold[v]]) + get_threshold(k, ufset.size_of(parent))
    #
    parent_index = np.array([i for i in range(num_node)])[np.array(ufset.size) > 1]
    print('end 1 parent_index.shape = ', parent_index.shape)
    print('end 1 parent_index = ', parent_index)
    print('end 1 ufset.parent = ', np.array(ufset.parent)[parent_index])
    print('end 1 ufset.size = ', np.array(ufset.size)[np.array(ufset.size) > 1])
    print('end 1 ufset.size.sum() = ', np.array(ufset.size)[np.array(ufset.size) > 1].sum())
    print('end 1 ufset.num_set = ', ufset.num_set)

    return ufset

def remove_small_component(ufset, sorted_graph, min_size, num_node):
    for edge in sorted_graph:
        u = ufset.find(edge[0])
        v = ufset.find(edge[1])

        if u != v:
            if ufset.size_of(u) < min_size or ufset.size_of(v) < min_size: ufset.merge(u, v)
    #
    parent_index = np.array([i for i in range(num_node)])[np.array(ufset.size) > 1]
    print('end 2 parent_index.shape = ', parent_index.shape)
    print('end 2 parent_index = ', parent_index)
    print('end 2 ufset.parent = ', np.array(ufset.parent)[parent_index])
    print('end 2 ufset.size = ', np.array(ufset.size)[np.array(ufset.size) > 1])
    print('end 2 ufset.size.sum() = ', np.array(ufset.size)[np.array(ufset.size) > 1].sum())
    print('end 2 ufset.num_set = ', ufset.num_set)

    return ufset


