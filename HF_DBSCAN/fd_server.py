import random
from flask import Flask, request
from flask_restful import Resource, Api
from typing import Dict, List

from utils import process_http_posts
from fd_dbscan import FDBSCAN_Server

def training(clients: List, server: FDBSCAN_Server, clients_selection_seed: int, missing_clients_percentage: int):
    import random
    random.seed(clients_selection_seed)
    n_clients = len(clients)
    n_clients_to_select = int(n_clients * (100 - missing_clients_percentage) / 100)
    selected_clients_idx = random.sample(range(n_clients), n_clients_to_select)
    selected_clients = [c for cli, c in enumerate(clients) if cli in selected_clients_idx]
    print('Training started')
    print(f'Selected clients: {selected_clients_idx}')

    data = {'action': 'compute_local_update'}
    local_updates, failures = process_http_posts(selected_clients, data)
    contribution_map = {}
    for i in range(len(local_updates)):
        local_update = local_updates[i]
        for string_key, value in local_update.items():
            tuple_key = eval(string_key)
            if tuple_key in contribution_map:
                contribution_map[tuple_key] += value
            else:
                contribution_map[tuple_key] = value

    cells, labels = server.compute_clusters(contribution_map)

    data = {'action': 'assign_points_to_cluster', 'cells': cells, 'labels': labels}
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

def run_server(host: str, port: int, MIN_POINTS: int, clients_selection_seed: int, missing_clients_percentage: int):

    clients = []
    client_ids = []

    params = {
        'MIN_POINTS': MIN_POINTS
    }
    server = FDBSCAN_Server()
    server.initialize(params)

    class FederatedServer(Resource):

        def get(self):
            action = request.args.get('action')
            if (action == 'start'):
                server.run();
                training(clients, server, clients_selection_seed, missing_clients_percentage)
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