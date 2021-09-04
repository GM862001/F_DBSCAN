from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from urllib.request import urlopen
import numpy as np
import pandas as pd
import json

from plot import plot2D

# results_folder = 'banana'
results_folder = 's-set1'
dataset_dir = '../datasets'
# dataset_file = 'banana.arff'
dataset_file = 's-set1.arff'
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

# MIN_POINTS = 4 # banana
MIN_POINTS = 15 # s-set1
# EPSILON = 0.03 # banana
EPSILON = 0.0325 # s-set1
clustering = DBSCAN(eps = EPSILON, min_samples = MIN_POINTS)
dbscan_labels = clustering.fit_predict(X)
dbscan_labels = [int(label) for label in dbscan_labels]
plot2D(points = X, labels = dbscan_labels, folder = results_folder, message = "DBSCAN")

start_port = 5000
N_clients = 10
url = f'http://localhost:8080/?action=start'
urlopen(url)
for port in range(start_port, start_port + N_clients):
	url = f'http://localhost:{port}/?action=results'
	response = urlopen(url)
	data_json = json.loads(response.read())
	result = np.array(data_json['results'])
	if len(result) == 0:
		continue
	if port == start_port:
		joined_result = result
	else:
		joined_result = np.concatenate((joined_result, result))
joined_points = joined_result[:,:2]
joined_labels = joined_result[:,2]
joined_labels = [int(label) for label in joined_labels]
plot2D(points = joined_points, labels = joined_labels, folder = results_folder, message = f'Federated DBSCAN')
# plot2D(points = joined_points, labels = joined_labels, folder = results_folder, message = f'Federated DBSCAN - Missing Client')
