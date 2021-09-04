import numpy as np
import pandas as pd
import uuid
from threading import Thread
from sklearn.preprocessing import MinMaxScaler
from scipy.io import arff

from fd_client import run_client

def generate_dataset_chunks(X: np.array, num_clients: int):
    dataset_chunks = []
    features_list = [i for i in range(len(X[0]))]
    for i in range(num_clients):
        dataset_chunks.append(X[:, features_list[i::num_clients]])
    return dataset_chunks

def prepare_dataset(num_clients: int):
    dataset_dir = '../datasets'
    # dataset_file = '3MC.arff'
    dataset_file = 'aggregation.arff'
    dataset_path = f'{dataset_dir}/{dataset_file}'
    dataset = arff.loadarff(dataset_path)
    df = pd.DataFrame(dataset[0])

    Y = df['class'].tolist()
    Y = np.array([-1 if y == b'noise' else int(y) for y in Y])
    del df['class']
    X_original = np.array(df.values)
    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(X_original)
    dataset_chunks = generate_dataset_chunks(X, num_clients)
    return dataset_chunks

server_url = "127.0.0.1:8080"
host = "127.0.0.1"
start_port = 5000
N_clients = 2
dataset_chunks = prepare_dataset(N_clients)

threads = []
for cli in range(N_clients):
    client_id = uuid.uuid4().hex 
    port = start_port + cli
    thread_obj = Thread(target = run_client, args = (client_id, server_url, host, port, dataset_chunks[cli]))
    threads.append(thread_obj)
    thread_obj.start()
    
for t in threads:
    t.join()