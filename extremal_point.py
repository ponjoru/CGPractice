import os
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt
from utils import is_above


class FileParser:
    keywords = ['Polygon', 'Directions']

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
        return np.array(data[0], dtype='float64'), np.array(data[1], 'float64')


class ExtremePoint:
    def __init__(self, polygon: np.array, direction: np.array, plot=False):
        """
        Find convex polygon extreme point in the given direction with complexity O(logN)
        :param polygon: 2d numpy array [[x1,y1], [x2, y2], ... , [xn, yn]]
        :param direction: 2d numpy array [[x1, y1], [x2, y2]]
        """
        self.polygon = polygon
        self.dir = direction
        self.n = polygon.shape[0]
        self.fig, self.ax = plt.subplots()

        if plot:
            self.plot_input_data()

    def plot_input_data(self):
        x = self.polygon[:, 0]
        y = self.polygon[:, 1]
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        self.ax.plot(x, y, 'b-')
        plt.show()

    def run(self):
        res1 = []
        res2 = []
        for dir in self.dir:
            dir = dir[1] - dir[0]
            x, y = self.find_extreme_point(dir)
            res1.append(x)
            res2.append(y)
        return res1, res2

    def find_extreme_point(self, dir):
        left = 0
        right = self.n - 1
        up_a = is_above(self.polygon[1], self.polygon[0], dir)
        up_b = is_above(self.polygon[self.n - 1], self.polygon[0], dir)
        if (not up_a) and (not up_b):
            return 0, self.polygon[0]

        while True:
            if right == left + 1:
                if is_above(self.polygon[right], self.polygon[left], dir):
                    return right, self.polygon[right]
                else:
                    return left, self.polygon[left]
            mid = (left + right) // 2
            up_mid = is_above(self.polygon[mid + 1], self.polygon[mid], dir)
            up_b = is_above(self.polygon[mid - 1], self.polygon[mid], dir)
            if (not up_mid) and (not up_b):
                return mid, self.polygon[mid]

            if up_a:
                if not up_mid:
                    right = mid
                else:
                    if is_above(self.polygon[left], self.polygon[mid], dir):
                        right = mid
                    else:
                        left = mid
                        up_a = up_mid
            else:
                if up_mid:
                    left = mid
                    up_a = up_mid
                else:
                    if is_above(self.polygon[mid], self.polygon[left], dir):
                        right = mid
                    else:
                        left = mid
                        up_a = up_mid


if __name__ == '__main__':
    filename = './extremal_point.txt'
    parser = FileParser(filename=filename)
    polygon, dirs = parser.parse()
    solver = ExtremePoint(polygon, dirs, plot=False)
    res1, res2 = solver.run()
    solution = zip(dirs, res2)
    for x, y in solution:
        print('direction:', x[1]-x[0], '; extremal point:', y)
