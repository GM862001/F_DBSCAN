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
        return self.__labels, self.__true_labels

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

        dense_cells = [tuple(row) for row in cells]
        self.__labels = [-1] * len(points)
        
        cell_points_dic = {}
        for actual_point in points:
            actual_cell = tuple(math.floor(actual_point[i] / self.__L) for i in range(len(actual_point)))
            try:
                cell_points_dic[actual_cell].append(actual_point)
            except: 
                cell_points_dic[actual_cell] = [actual_point]

        cells = list(cell_points_dic.keys())
                        
        dense_cells_index = dense_cells.index
        points_index = points.index
        for cell in cells:
            current_cell_points = cell_points_dic[cell]
            if (cell in dense_cells):
                cluster = labels[dense_cells_index(cell)]
                self.__assign_labels_to_points(points_index, current_cell_points, cluster)
            else:
                nearest_adjacent_cell = self.__locate_nearest_adjacent_cell(cell, dense_cells, current_cell_points)                            
                if (nearest_adjacent_cell is not None):
                    cluster = labels[dense_cells_index(nearest_adjacent_cell)]
                    self.__assign_labels_to_points(points_index, current_cell_points, cluster)
        
    def __locate_nearest_adjacent_cell(self, cell: Tuple, dense_cells: List, current_cell_points: List):
        adjacent_cells = get_all_neighbors(cell)
        selected_adjacent_cell = None
        min_dist = float('inf')
        for adjacent_cell in adjacent_cells:
            if (adjacent_cell in dense_cells):
                adjacent_cell_mid_point = tuple(cell_coord * self.__L + self.__L/2 for cell_coord in adjacent_cell)
                current_min_dist = min([distance.euclidean(adjacent_cell_mid_point, cell_point) for cell_point in current_cell_points])
                if (current_min_dist < min_dist):
                    selected_adjacent_cell = adjacent_cell
        return selected_adjacent_cell
        
    def __assign_labels_to_points(self, points_index: Callable, cell_points: List, cluster: int):
        labels = self.__labels
        for point in cell_points:
            labels[points_index(point)] = cluster

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
