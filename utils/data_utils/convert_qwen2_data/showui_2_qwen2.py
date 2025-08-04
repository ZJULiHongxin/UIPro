
import os, json, cv2
from tqdm import tqdm

DATA_DIR = "/mnt/vdb1/juzheng/data/ShowUI"

file = os.path.join(DATA_DIR, "mutilturn_showui.json")

data = json.load(open(file))

for x in tqdm(data, total=len(data)):
    

