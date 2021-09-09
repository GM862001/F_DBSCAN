import math
import numpy as np
from typing import List, Dict, Tuple, Callable
from scipy.spatial import distance

def get_all_neighbors(cell: tuple):
    diag_coord = [(x - 1, x, x + 1) for x in cell]
    cartesian_product = [[]]
    for pool in diag_coord:
        cartesian_product = [(x + [y]) for x in cartesian_product for y in pool]

    neighbors = []
    for prod in cartesian_product:
        differential_coord = 0
        for i in range(len(prod)):
            if prod[i] != cell[i]:
                differential_coord += 1
        if differential_coord == 1:
            neighbors.append(tuple(prod))
    return neighbors

class FDBSCAN_Client():

    def initialize(self, params: Dict):
        self.__dataset = params['dataset']
        self.__L = params['L']
        self.__labels = []
        self.__true_labels = params['true_labels']
        self.__passive = True

    def get_dataset(self):
        return self.__dataset

    def get_labels(self):
        return self.__labels, self.__true_labels, self.__get_points()

    def is_passive(self):
        return self.__passive

    def __get_points(self, floor: bool = False):
        dimension = len(self.__dataset[0])
        points = []
        for row in self.__dataset:
            if floor:
                points.append(tuple(math.floor(row[i] / self.__L) for i in range(dimension)))
            else:
                points.append(tuple(row[i] for i in range(dimension)))
        return points

    def compute_local_update(self):

        self.__passive = False

        cells = np.array(self.__get_points(floor = True))
        dimensions = len(cells[0])

        max_cell_coords = []
        min_cell_coords = []
        for i in range(dimensions):
            max_cell_coords.append(np.amax(cells[:, i]))
            min_cell_coords.append(np.amin(cells[:, i]))

        shifts = np.zeros(dimensions)
        for i in range(dimensions):
            if min_cell_coords[i] < 0:
                shifts[i] = -1 * min_cell_coords[i]

        shifted_dimensions = ()
        for i in range(dimensions):
            shifted_dimensions += (int(max_cell_coords[i] + 1 + shifts[i]), )

        count_matrix = np.zeros(shifted_dimensions)
        for cell in cells:
            shifted_cell_coords = ()
            for i in range(dimensions):
                shifted_cell_coords += (int(cell[i] + shifts[i]),)
            count_matrix[shifted_cell_coords] += 1

        non_zero = np.where(count_matrix > 0)
        non_zero_indexes = []
        for i in range(len(non_zero)):
            for j in range(len(non_zero[i])):
                if i == 0:
                    non_zero_indexes.append((int(non_zero[i][j]), ))
                else:
                    non_zero_indexes[j] += (int(non_zero[i][j]), )

        dict_to_return = {}
        for index in non_zero_indexes:
            shifted_index = ()
            for i in range(len(index)):
                shifted_index += (int(index[i] - shifts[i]), )
            dict_to_return[shifted_index] = count_matrix[index]

        return dict_to_return

    def assign_points_to_cluster(self, cells: List, labels: List):

        points = self.__get_points()

        dense_cells = []
        for row in cells:
            dense_cells.append(tuple(row))

        while len(points) > 0:
            actual_point = points.pop(0)
            actual_cell = tuple(math.floor(actual_point[i] / self.__L) for i in range(len(actual_point)))
            outlier = True
            if actual_cell in dense_cells:
                self.__labels.append(labels[dense_cells.index(actual_cell)])
            else:
                min_dist = float('inf')
                cluster_to_assign = -1
                check_list = get_all_neighbors(actual_cell)
                for check_cell in check_list:
                    if check_cell in dense_cells:
                        cell_mid_point = tuple(cell_coord * self.__L + self.__L/2 for cell_coord in check_cell)
                        actual_dist = distance.euclidean(actual_point, cell_mid_point)
                        if actual_dist < min_dist:
                            min_dist = actual_dist
                            cluster_to_assign = labels[dense_cells.index(check_cell)]
                        outlier = False
                self.__labels.append(cluster_to_assign)

        self.__training_completed = True

class FDBSCAN_Server():

    def initialize(self, params: Dict):
        self.__MIN_POINTS = params['MIN_POINTS']
        self.__running = False

    def get_running(self):
        return self.__running

    def run(self, value: bool = True):
        self.__running = value

    def compute_clusters(self, contribution_map: Dict):

        key_list = list(contribution_map.keys())
        value_list = list(contribution_map.values())

        n_cells = len(key_list)
        visited = np.zeros(n_cells)
        clustered = np.zeros(n_cells)
        cells = []
        labels = []
        cluster_ID = 0

        while 0 in visited:
            curr_index = np.random.choice(np.where(np.array(visited) == 0)[0])
            curr_cell = key_list[curr_index]
            visited[curr_index] = 1

            num_points = value_list[curr_index]
            if num_points >= self.__MIN_POINTS:
                cells.append(curr_cell)
                labels.append(cluster_ID)
                clustered[curr_index] = 1

                list_of_cells_to_check = get_all_neighbors(curr_cell)
                while len(list_of_cells_to_check) > 0:
                    neighbor = list_of_cells_to_check.pop(0)
                    neighbor_index = key_list.index(neighbor) if neighbor in key_list else ""
                    if neighbor in key_list and visited[neighbor_index] == 0:
                        visited[neighbor_index] = 1
                        if value_list[neighbor_index] >= self.__MIN_POINTS:
                            list_of_cells_to_check += get_all_neighbors(neighbor)
                        if clustered[neighbor_index] == 0:
                            cells.append(neighbor)
                            labels.append(cluster_ID)
                            clustered[neighbor_index] = 1
                cluster_ID += 1

        return cells, labels
