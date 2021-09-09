from fd_server import run_server

host = "127.0.0.1"
port = 8080

MIN_POINTS = 4 # 3MC
# MIN_POINTS = 6 # aggregation
EPSILON = 0.1 # 3MC
# EPSILON = 0.04 # aggregation

run_server(host, port, MIN_POINTS, EPSILON)