from typing import List
from flask import Flask, request
from flask_restful import Resource, Api

from utils import send_post
from fd_dbscan import FDBSCAN_Client

def run_client(client_id: str, server_url: str, host: str, port: int, dataset: List, L: float):

    address = host + f':{port}'
    send_post(server_url, {'action': 'connect','client_id': client_id, 'address': address})

    params = {
        'dataset': dataset,
        'L': L
    }
    client = FDBSCAN_Client()
    client.initialize(params)

    class FederatedClient(Resource):
    
        def get(self):
            action = request.args.get('action')
            if (action == 'dataset'):
                dataset = client.get_dataset().tolist()
                return {'rows': len(dataset), 'dataset': dataset}
            elif (action == 'results'):
                results = client.get_results();
                if len(results) > 0:
                    return {'results': results.tolist()}
                else:
                    return {'message': 'Training not completed', 'results': []}
            else:
                return {'message': 'Hello World from Client', 'client_id': client_id}
    
        def post(self):
            json_data = request.json
            action = json_data.get('action')
            if (action == 'compute_local_update'):
                result = client.compute_local_update()
                to_return = {}
                for tuple_key in result:
                    string_key = ','.join([str(coord) for coord in tuple_key])
                    to_return[string_key] = result[tuple_key]
                return to_return
            elif (action == 'assign_points_to_cluster'):
                cells = json_data.get('cells')
                labels = json_data.get('labels')
                client.assign_points_to_cluster(cells, labels)
            else:
                return {'message': 'Action not Supported'}

    app = Flask(client_id)
    api = Api(app)
    api.add_resource(FederatedClient, '/')
    app.run(host = host, port = port)