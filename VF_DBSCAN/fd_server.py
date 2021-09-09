import numpy as np
from flask import Flask, request
from flask_restful import Resource, Api
from typing import Dict, List

from utils import process_http_posts
from fd_dbscan import FDBSCAN_Server

def training(clients: List, server: FDBSCAN_Server):
    data = {'action': 'compute_neighborhood_matrix', 'epsilon': server.get_epsilon()}
    results, failures = process_http_posts(clients, data)

    N = len(results[0]['matrix'])
    global_matrix = np.zeros((N, N))
    for i in range(len(results)):
        matrix = np.array(results[i]['matrix'])
        global_matrix += matrix
    global_matrix = np.where(global_matrix < len(results), 0, 1)

    Q = server.DBSCAN(global_matrix)

    data = {'action': 'update_labels', 'labels': Q.tolist()}
    process_http_posts(clients, data)

def connect(json_data: Dict, clients: List, client_ids: List, running: bool):
    if (not running):
        client_id = json_data.get('client_id')
        address = json_data.get('address')
        client_ids.append(client_id)
        clients.append(address)
        code = 200
        message = {'message': f'Client {client_id} Connected'}
    else:
        code = 500
        message = {'message': 'Unable to connect: a training process is runnning'}
    return message, code

def run_server(host: str, port: int, MIN_POINTS: int, EPSILON: float):

    clients = []
    client_ids = []

    params = {
        'MIN_POINTS': MIN_POINTS,
        'EPSILON': EPSILON
    }
    server = FDBSCAN_Server()
    server.initialize(params)

    class FederatedServer(Resource):

        def get(self):
            action = request.args.get('action')
            if (action == 'start'):
                server.run();
                training(clients, server)
                server.run(False);
                return {'message': 'Training completed'}
            else:
                return {'message': 'Hello world from server', 'clients': clients}
    
        def post(self):
            json_data = request.get_json()
            action = json_data.get('action')
            if (action == 'connect'):
                return connect(json_data, clients, client_ids, server.get_running())
            else:
                return {'message': 'Action not Supported'}, 201

    app = Flask(__name__)
    api = Api(app)
    api.add_resource(FederatedServer, '/')
    app.run(host = host, port = port)