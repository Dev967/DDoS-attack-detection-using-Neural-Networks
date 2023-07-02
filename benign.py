import csv
import functools
import random
import time
from argparse import ArgumentParser
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

parser = ArgumentParser(
    prog='Benign Simulator',
    description='Simulates Benign requests for our exeperiment',
    epilog='just a script')

parser.add_argument('host')
args = parser.parse_args()
host = args.host

pages = ["/index.html", "/about.html", "/gallery.html", "/login.html", "/admin"]
out_file = open(f'benign_output.csv', "w")
csv_writer = csv.writer(out_file)
csv_writer.writerow(["TIMESTAMP", "URL", "START_TIME", "END_TIME", "EALPSED_TIME", "STATUS", "METHOD"])


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        rotue, status, method = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        timestamp = datetime.now(tz=ZoneInfo('Asia/Kolkata'))
        row = [str(timestamp), route, str(tic), str(toc), str(elapsed_time), str(status), method]
        csv_writer.writerow(row)

    return wrapper_timer


@timer
def execute_request(session, route):
    try:
        resp = session.get(f'http://{host}{route}', timeout=30)
    except:
        return route, 500, "GET"
    return route, resp.status_code, "GET"


@timer
def execute_login_request(session):
    try:
        resp = session.post(f'http://{host}/login', {
            "username": "admin",
            "password": "password@123"
        }, timeout=30)
    except:
        return "/login", 500, "POST"
    return "/login", resp.status_code, "POST"


while True:
    s = requests.session()

    for i in range(5):
        idx = random.randrange(0, 6)
        if idx == 5: break
        route = pages[idx]
        if route == "/login.html":
            execute_request(s, route)
            if random.randrange(0, 2) > 0: execute_login_request(s)  # lower chances by 50%
        else:
            execute_request(s, route)
        time.sleep(random.randrange(1, 3))

    delay = random.randrange(1, 11)
    time.sleep(delay)
