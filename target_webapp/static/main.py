import random
import time

# resp = requests.get("http://192.168.1.16/")
# print(resp.status_code)

# home, about, gallery, login, admin

pages = ["home", "about", "gallery", "login", "admin"]
while True:

    for i in range(5):
        idx = random.randrange(0, 6)
        if idx == 5: break
        route = pages[idx]

    next_req_delay = random.randrange(1, 11)
    time.sleep(next_req_delay)
