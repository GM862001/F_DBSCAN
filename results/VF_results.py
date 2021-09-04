from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from urllib.request import urlopen
import numpy as np
import pandas as pd
import json

from plot import plot2D

# results_folder = '3MC'
results_folder = 'aggregation'
dataset_dir = '../datasets'
# dataset_file = '3MC'
dataset_file = 'aggregation.arff'
dataset_path = f'{dataset_dir}/{dataset_file}'
dataset = arff.loadarff(dataset_path)
df = pd.DataFrame(dataset[0])

true_labels = df['class'].tolist()
true_labels = np.array([-1 if label == b'noise' else int(label) for label in true_labels])
del df['class']
X_original = np.array(df.values)
min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X_original)
plot2D(points = X, labels = true_labels, folder = results_folder, message = "Ground Truth")

# MIN_POINTS = 4 # 3MC
MIN_POINTS = 6 # aggregation
# EPSILON = 0.1 # 3MC
EPSILON = 0.04125 # aggregation
clustering = DBSCAN(eps = EPSILON, min_samples = MIN_POINTS)
dbscan_labels = clustering.fit_predict(X)
dbscan_labels = [int(label) for label in dbscan_labels]
plot2D(points = X, labels = dbscan_labels, folder = results_folder, message = "DBSCAN")

start_port = 5000
N_clients = 2
url = f'http://localhost:8080/?action=start'
urlopen(url)
for port in range(start_port, start_port + N_clients):
	url = f'http://localhost:{port}/?action=results'
	response = urlopen(url)
	data_json = json.loads(response.read())
	result = np.array(data_json['results'])
	if port == start_port:
		joined_result = result
	else:
		joined_result = np.insert(joined_result, 1, result[:, 0], axis = 1)
joined_points = joined_result[:,:2]
joined_labels = joined_result[:,2]
joined_labels = [int(label) for label in joined_labels]
plot2D(points = joined_points, labels = joined_labels, folder = results_folder, message = f'Federated DBSCAN')