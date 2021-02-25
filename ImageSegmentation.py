import sys
import cv2
import random as rand
import time
import numpy as np
import GraphOperator as go

def generate_image(ufset, width, height):
    random_color = lambda: (int(rand.random() * 255), int(rand.random() * 255), int(rand.random() * 255))
    color = [random_color() for i in range(width * height)]

    save_img = np.zeros((height, width, 3), np.uint8)

    color_index = []

    for y in range(height):
        for x in range(width):
            color_idx = ufset.find(y * width + x)
            color_index.append(color_idx)
            save_img[y, x] = color[color_idx]
    print('color_index = ', set(color_index))
    return save_img

#
k = 1
min_size = 50
#读取图片
img = cv2.imread('1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width, channel = img.shape#height=220,width=249
print('img.shape = ', img.shape, 'height = ', height, 'width = ', width, 'channel = ', channel)
img = np.asarray(img, dtype=float)
#分开组成grb
img = cv2.split(img)
#建立图结构
graph = go.build_graph(img, width, height)
#按照权重进行不减的排序
weight = lambda edge: edge[2]
sorted_graph = sorted(graph, key=weight)#根据权重对所有的边进行排序
#分割
ufset = go.segment_graph(sorted_graph, width * height, k)
ufset = go.remove_small_component(ufset, sorted_graph, min_size, width * height)
#可视化结果
save_img = generate_image(ufset, width, height)#
#保存结果
cv2.imwrite('1_result.png', save_img)



