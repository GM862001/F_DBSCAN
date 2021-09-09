import numpy as np
from typing import List, Dict
from scipy.spatial import distance
from numba import jit
import time

@jit(nopython=True)
def numba_euclidean_distance(u:np.ndarray, v:np.ndarray):
    return np.linalg.norm(u - v)

class FDBSCAN_Client():

    def initialize(self, params: Dict):
        self.__dataset = params['dataset']
        self.__labels = []

    def get_results(self):
        return self.__labels

    def __get_points(self):
        dimension = len(self.__dataset[0])
        points = []
        for row in self.__dataset:
            points.append(tuple(row[i] for i in range(dimension)))
        return points

    def compute_neighborhood_matrix(self, epsilon: float):
        points = np.array(self.__get_points())
        n_points = len(points)
        matrix = [[0] * n_points for i in range(n_points)]
        start = time.time()
        for i in range(n_points):
            for j in range(n_points):
                if (numba_euclidean_distance(points[i], points[j]) < epsilon):
                    matrix[i][j] = 1
        end = time.time()
        print(f'compute_neighborhood_matrix {end - start}')
        return matrix

    def update_labels(self, labels: List):
        self.__labels = labels

class FDBSCAN_Server():

    def initialize(self, params: Dict):
        self.__MIN_POINTS = params['MIN_POINTS']
        self.__EPSILON = params['EPSILON']
        self.__running = False

    def get_running(self):
        return self.__running

    def run(self, value: bool = True):
        self.__running = value

    def get_epsilon(self):
        return self.__EPSILON

    def DBSCAN(self, global_matrix: np.ndarray):
        N = len(global_matrix)
        visited = np.zeros(N)
        labels = np.zeros(N)
        labels -= 1
        cluster_ID = 0
        for i in range(N):
            if visited[i]:
                continue
            else:
                visited[i] = 1
                num_points = np.sum(global_matrix[i])
                if num_points >= self.__MIN_POINTS:
                    labels[i] = cluster_ID
                    to_visit = np.where(global_matrix[i] == 1)[0].tolist()
                    cluster_ID = self.__expand_cluster(to_visit, visited, global_matrix, labels, cluster_ID)
        return labels
    
    def __expand_cluster(self, to_visit: List, visited: np.array, global_matrix:np.ndarray, labels: np.array, cluster_ID: int):
        for j in to_visit:
            if visited[j] == 0:
                visited[j] = 1
                num_neighbors = np.sum(global_matrix[j])
                if num_neighbors >= self.__MIN_POINTS:
                    to_visit += np.where(global_matrix[j] == 1)[0].tolist()
            if labels[j] == -1:
                labels[j] = cluster_ID
        cluster_ID += 1
        return cluster_ID
