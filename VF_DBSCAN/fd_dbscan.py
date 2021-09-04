import numpy as np
from typing import List, Dict
from scipy.spatial import distance

class FDBSCAN_Client():

    def initialize(self, params: Dict):
        self.__dataset = params['dataset']
        self.__labels = []
        self.__training_completed = False

    def get_dataset(self):
        return self.__dataset

    def get_results(self):
        if self.__training_completed:
            dataset = np.array(self.__dataset)
            labels = np.array([self.__labels])
            return np.concatenate((dataset, labels.T), axis = 1)
        else:
            return []

    def get_points(self):
        dimension = len(self.__dataset[0])
        points = []
        for row in self.__dataset:
            points.append(tuple(row[i] for i in range(dimension)))
        return points

    def compute_neighborhood_matrix(self, epsilon: float):
        points = self.get_points()
        matrix = []
        for i in range(len(points)):
            row = []
            for j in range(len(points)):
                if distance.euclidean(points[i], points[j]) <= epsilon:
                    row.append(1)
                else:
                    row.append(0)
            matrix.append(row)
        return matrix

    def update_labels(self, labels: List):
        self.__training_completed = True
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
                    for j in to_visit:
                        if visited[j] == 0:
                            visited[j] = 1
                            num_neighbors = np.sum(global_matrix[j])
                            if num_neighbors >= self.__MIN_POINTS:
                                to_visit += np.where(global_matrix[j] == 1)[0].tolist()
                        if labels[j] == -1:
                            labels[j] = cluster_ID
                    cluster_ID += 1

        return labels