from fd_server import run_server

host = "127.0.0.1"
port = 8080
MIN_POINTS = 4 # banana
# MIN_POINTS = 15 # s-set1

clients_selection_seed = 1
missing_client_percentage = 0

run_server(host, port, MIN_POINTS, clients_selection_seed, missing_client_percentage)
