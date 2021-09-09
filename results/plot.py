from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot2D(points: np.ndarray, labels: np.array, folder: str, message: str):
    int_labels = [int(label) for label in labels]
    color_range = cm.rainbow(np.linspace(0, 1, np.max(np.unique(int_labels))+1))
    colors = []
    count_outliers = 0

    for label in int_labels:
        if label == -1:
            count_outliers += 1
            colors.append([0, 0, 0, 1])
        else:
            colors.append(color_range[label])

    plt.scatter(points[:, 0], points[:, 1], color = colors, marker = "o")
    plt.title(f'{message} - '
              f'{len(int_labels)} Points - '
              f'{len(np.unique(int_labels)) - (1 if count_outliers > 0 else 0)} Clusters - '
              f'{count_outliers} Outliers')
    plt.savefig(f'{folder}/{message}.png')
    plt.clf()