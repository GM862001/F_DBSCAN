from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from urllib.request import urlopen
import numpy as np
import pandas as pd
import json

from plot import plot2D
import metrics as mtr

results_folder = 'banana'
# results_folder = 's-set1'

start_port = 5000
N_clients = 10
url = 'http://localhost:8080/?action=start'
urlopen(url)

dataset = []
active_dataset = []
passive_dataset = []
labels = []
active_labels = []
passive_labels = []
true_labels = []
active_true_labels = []
passive_true_labels = []
passive_clients = False
for port in range(start_port, start_port + N_clients):
	url = f'http://localhost:{port}/?action=results'
	response = urlopen(url)
	data_json = json.loads(response.read())
	res_dataset = data_json['dataset']
	res_labels = data_json['labels']
	res_true_labels = data_json['true_labels']
	passive = data_json['passive']
	dataset += res_dataset
	labels += res_labels
	true_labels += res_true_labels
	if passive:
		passive_clients = True
		passive_dataset += res_dataset
		passive_labels += res_labels
		passive_true_labels += res_true_labels
	else:
		active_dataset += res_dataset
		active_labels += res_labels
		active_true_labels += res_true_labels

MIN_POINTS = 4 # banana
# MIN_POINTS = 15 # s-set1
EPSILON = 0.03 # banana
# EPSILON = 0.03 # s-set1
clustering = DBSCAN(eps = EPSILON, min_samples = MIN_POINTS)
dbscan_labels = clustering.fit_predict(np.array(dataset))

plot2D(points = np.array(dataset), labels = np.array(true_labels), folder = results_folder, message = 'Ground Truth')
plot2D(points = np.array(dataset), labels = dbscan_labels, folder = results_folder, message = 'DBSCAN')
plot2D(points = np.array(dataset), labels = np.array(labels), folder = results_folder, message = 'Federated DBSCAN')
plot2D(points = np.array(active_dataset), labels = np.array(active_labels), folder = results_folder, message = 'Federated DBSCAN - Active')
if passive_clients:
	plot2D(points = np.array(passive_dataset), labels = np.array(passive_labels), folder = results_folder, message = 'Federated DBSCAN - Passive')

mtr.print_metrics(true_labels, dbscan_labels, 'DBSCAN')
mtr.print_metrics(true_labels, labels, 'Federated DBSCAN')
mtr.print_metrics(active_true_labels, active_labels, 'Federated DBSCAN - Active')
if passive_clients:
	mtr.print_metrics(passive_true_labels, passive_labels, 'Federated DBSCAN - Passive')