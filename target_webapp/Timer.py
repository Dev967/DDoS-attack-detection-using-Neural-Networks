import functools
import time
from datetime import datetime
from zoneinfo import ZoneInfo


# filename = "out_file.csv"
# out_file = open(filename, 'w')
# csv_writer = csv.writer(out_file)
# csv_writer.writerow(['TIMESTAMP', 'SRC', 'URL', 'START_TIME', 'END_TIME', 'ELAPSED_TIME', 'STATUS'])


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        r, url, ip, status = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        timestamp = datetime.now(tz=ZoneInfo('Asia/Kolkata'))
        row = [str(timestamp), ip, url, str(toc), str(tic), str(elapsed_time), str(status)]
        print(row)
        # csv_writer.writerow(row)
        return r

    return wrapper_timer
