import requests
import concurrent.futures
from typing import Dict, List

def send_post(url: str, data: Dict):
    r = requests.post(f'http://{url}', json = data)
    return r.json(), r.status_code

def process_http_posts(clients: List, data: Dict):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(send_post, c, data) for c in clients]
        concurrent.futures.wait(futures)

    results = []
    failures = []

    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result[0])

    return results, failures
