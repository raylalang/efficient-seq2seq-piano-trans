from datetime import datetime
from tqdm import tqdm
import os
import threading
import time

list_to_sync = [
    "runs/wandb/offline-5gbsnfmc/run-5gbsnfmc.wandb",
]

interval = 60
count = 0
def sync():
    global interval
    global count
    print(count)
    if count == 0:
        for log in tqdm(list_to_sync, desc="wandb sync"):
            os.system("wandb sync %s"%log)
    time.sleep(1)
    count -= 1
    if count < 0:
        count = interval

while 1:
    sync()

