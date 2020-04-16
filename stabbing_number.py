import os
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from collections import deque
from utils import counting_sort


class FileParser:
    keywords = ['Polygon', 'Triangulation triangles']

    def __init__(self, filename):
        self.filename = filename
        if not os.path.isfile(self.filename):
            raise FileNotFoundError

    def parse(self):
        data = []
        with open(self.filename, 'r') as file:
            line = next(file)
            while line:
                keyword = [keyword for keyword in self.keywords if line.startswith(keyword)][0]
                if keyword:
                    split = line.split(' = ')
                    data.append(literal_eval(split[-1]))
                try:
                    line = next(file)
                except StopIteration:
                    line = ''
        return np.array(data[0]), np.array(data[1])


class StabbingNumber:
    def __init__(self, polygon=None, triangles_arr=None, filename=None, plot=False):
        """
        Find stabbing number of given triangulation of a convex polygon with complexity O(N)
        Stabbing number - the maximal number of intersections between a line and edges of the triangulation.
        :param polygon: numpy array of shape [n, 2] - N (x,y) points
        :param triangles_arr: numpy array of shape [n-2,3] - N-2 triangles defined as array of 3 vertex ids
        :param filename:
        """
        self.filename = filename
        if filename:
            print("Parsing file ", filename)
            self.file_parser = FileParser(filename)
            self.polygon, self.triangles_arr = self.file_parser.parse()
        else:
            self.polygon = polygon
            self.triangles_arr = triangles_arr
        if not self.triangles_arr.shape[0] == self.polygon.shape[0] - 2:
            raise ValueError
        self.fig, self.ax = plt.subplots()

        self.triangulation_graph = self.init_graph()

        if plot:
            self.plot_input_data()

    def init_graph(self):
        graph = []
        for i in range(self.triangles_arr.shape[0]):
            graph.append([])
        return graph

    def build_triangulation_graph(self):
        """
        Build graph of the triangulation. Vertices correspond to the triangles of triangulation. Two triangles
        sharing an edge are connected in the triangulation graph
        :return:
        """
        arr = []
        # build array of (x,y,i), where x:y - represents a triangle edge, x - id of start point, y - id of end point
        # i - the triangle id
        for i, triangle in enumerate(self.triangles_arr):
            keys = [(x, y, i) for x in triangle for y in triangle if x < y]  # O(9) a.k.a O(1)
            for key in keys:
                arr.append(key)

        # lexicographical sorting the array of tuples by its keys: y, then x
        arr = counting_sort(arr, 1)
        arr = counting_sort(arr, 0)

        # if two neighbours in sorted array are equal then the corresponding triangles are connected in the graph
        for i in range(len(arr)-1):
            if arr[i][:-1] == arr[i+1][:-1]:
                self.triangulation_graph[arr[i][-1]].append(arr[i+1][-1])
                self.triangulation_graph[arr[i+1][-1]].append(arr[i][-1])

    def bfs(self, source: int):
        """
        breadth-first search implementation
        :param source: point id to start searching from
        :return: distance array to all the vertices from the source
        """
        d = [0] * len(self.triangulation_graph)
        d[source] = 0

        # deque should be used, pop left operation O(1) should be used to obtain true O(N) complexity in general
        q = deque()
        q.append(source)

        visited = [False] * (len(self.triangulation_graph))
        visited[source] = True

        while q:
            s = int(q.popleft())
            for i in self.triangulation_graph[s]:
                if not visited[i]:
                    d[i] = d[s] + 1
                    visited[i] = True
                    q.append(i)
        return np.array(d)
        # O(V+E) ~= O(N-2 + N-3) ~= O(N)

    def find_tree_diameter(self):
        v = 0
        d = self.bfs(v)                             # O(N)
        u = int(np.argmax(d))                       # O(N)
        d = self.bfs(u)                             # O(N)
        return max(d)
        # O(N)

    def run(self):
        self.build_triangulation_graph()                    # O(N)
        diameter = self.find_tree_diameter()                # O(N)
        return diameter + 2
        # O(N)

    def plot_input_data(self):
        x = self.polygon[:, 0]
        y = self.polygon[:, 1]
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        self.ax.plot(x, y, 'b-')
        for i, triangle in enumerate(self.triangles_arr):
            sub_x = []
            sub_y = []
            for j in triangle:
                sub_x.append(self.polygon[j][0])
                sub_y.append(self.polygon[j][1])
            sub_x.append(sub_x[0])
            sub_y.append(sub_y[0])
            self.ax.plot(sub_x, sub_y, 'r-')
        plt.show()

# 1: 4
# Polygon = [[0, 0], [1, 1], [2, 4], [0.5, 7], [-2, 6], [-4, 2]]
# Triangulation triangles = [[0, 1, 5], [1, 2, 3], [3, 4, 5], [1, 3, 5]]

# 2: 4
# Polygon = [[0, 0], [1, 1], [2, 4], [3, 9], [4,16], [5,25]]
# Triangulation triangles = [[0, 2, 4], [2, 0, 1], [4, 2, 3], [0, 4, 5]]

# 3: 3
# Polygon = [[0, 0], [1, 1], [2, 4], [3, 9]]
# Triangulation triangles = [[0, 1, 3], [3, 1, 2]]


# script evaluation
solver = StabbingNumber(filename='./stabbing_number.txt', plot=False)
result = solver.run()
print('Stabbing Number of given triangulation is: ', result)


