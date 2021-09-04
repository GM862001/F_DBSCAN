from fd_server import run_server

host = "127.0.0.1"
port = 8080
# MIN_POINTS = 4 # banana
MIN_POINTS = 15 # s-set1

run_server(host, port, MIN_POINTS)