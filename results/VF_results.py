from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from urllib.request import urlopen
import numpy as np
import pandas as pd
import json

from plot import plot2D
import metrics as mtr

results_folder = '3MC'
# results_folder = 'aggregation'

dataset_dir = '../datasets'
dataset_file = 'banana.arff'
# dataset_file = 'aggregation.arff'
dataset_path = f'{dataset_dir}/{dataset_file}'
dataset = arff.loadarff(dataset_path)
df = pd.DataFrame(dataset[0])

true_labels = df['class'].tolist()
true_labels = np.array([-1 if label == b'noise' else int(label) for label in true_labels])
del df['class']
X_original = np.array(df.values)
min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X_original)

MIN_POINTS = 4 # 3MC
# MIN_POINTS = 6 # aggregation
EPSILON = 0.1 # 3MC
# EPSILON = 0.04 # aggregation
clustering = DBSCAN(eps = EPSILON, min_samples = MIN_POINTS)
dbscan_labels = clustering.fit_predict(X)

url = 'http://localhost:8080/?action=start'
urlopen(url)
url = 'http://localhost:5000/?action=results'
response = urlopen(url)
data_json = json.loads(response.read())
labels = np.array(data_json['labels'])

plot2D(points = X, labels = true_labels, folder = results_folder, message = "Ground Truth")
plot2D(points = X, labels = dbscan_labels, folder = results_folder, message = "DBSCAN")
plot2D(points = X, labels = labels, folder = results_folder, message = f'Federated DBSCAN')

mtr.print_metrics(true_labels, dbscan_labels, 'DBSCAN')
mtr.print_metrics(true_labels, labels, 'Federated DBSCAN')
