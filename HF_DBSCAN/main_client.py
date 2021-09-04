import numpy as np
import pandas as pd
import uuid
from typing import List
from threading import Thread
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from scipy.io import arff

from fd_client import run_client

def generate_dataset_chunks(X: np.array, Y: List, n_splits: int, shuffle: bool = True):
    if (n_splits == 1):
        return [X]
    skf = StratifiedKFold(n_splits = n_splits, shuffle = shuffle)
    dataset_chunks = []
    for train_index, test_index in skf.split(X, Y):
        dataset_chunks.append(X[test_index])
    return dataset_chunks

def prepare_dataset(num_clients: int):
    dataset_dir = '../datasets'
    # dataset_file = 'banana.arff'
    dataset_file  = 's-set1.arff'
    dataset_path = f'{dataset_dir}/{dataset_file}'
    dataset = arff.loadarff(dataset_path)
    df = pd.DataFrame(dataset[0])

    Y = df['class'].tolist()
    Y = np.array([-1 if y == b'noise' else int(y) for y in Y])
    del df['class']
    X_original = np.array(df.values)
    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(X_original)
    dataset_chunks = generate_dataset_chunks(X, Y, num_clients)
    return dataset_chunks

server_url = "127.0.0.1:8080"
host = "127.0.0.1"
start_port = 5000
N_clients = 10
dataset_chunks = prepare_dataset(N_clients)
# L = 0.03 # banana
L = 0.03 # s-set1

threads = []
for cli in range(N_clients):
    client_id = uuid.uuid4().hex 
    port = start_port + cli
    thread_obj = Thread(target = run_client, args = (client_id, server_url, host, port, dataset_chunks[cli], L))
    threads.append(thread_obj)
    thread_obj.start()
    
for t in threads:
    t.join()