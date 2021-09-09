from typing import List
from flask import Flask, request
from flask_restful import Resource, Api

from utils import send_post
from fd_dbscan import FDBSCAN_Client

def run_client(client_id: str, server_url: str, host: str, port: int, dataset: List):

    address = host + f':{port}'
    send_post(server_url, {'action': 'connect','client_id': client_id, 'address': address})

    params = {
        'dataset': dataset,
    }
    client = FDBSCAN_Client()
    client.initialize(params)

    class FederatedClient(Resource):
    
        def get(self):
            action = request.args.get('action')
            if (action == 'results'):
                labels = client.get_results();
                return {'labels': labels}
            else:
                return {'message': 'Hello World from Client', 'client_id': client_id}

        def post(self):
            json_data = request.json
            action = json_data.get('action')
            if (action == 'compute_neighborhood_matrix'):
                epsilon = json_data.get('epsilon')
                neighborhood_matrix = client.compute_neighborhood_matrix(epsilon)
                return {'matrix': neighborhood_matrix}
            elif (action == 'update_labels'):
                labels = json_data.get('labels')
                client.update_labels(labels)
            else:
                return {'message': 'Action not Supported'}, 201

    app = Flask(client_id)
    api = Api(app)
    api.add_resource(FederatedClient, '/')
    app.run(host = host, port = port)