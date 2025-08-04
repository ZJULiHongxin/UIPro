import os, json, random
from copy import deepcopy
from tqdm import tqdm

index = 2
path = ["/mnt/nvme1n1p1/hongxin_li/highres_autogui_data/vg_refcoco_g_plus_5725059QAs.json",
        "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/L20_seeclick_PtBoxGndRef_widcap_ricosca_refexp_motif_web_869k_llava_merged_459936_MAXTOK650.json",
        "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/autogui625k_merged_android_77k.json"
    ][index]
data = json.load(open(path))

scalings = [
    [1600, 8e3, 4e4, 2e5, 1e6, 5e6],
    [8469, 42348, 211741, 1058705, 5293525],
    [25000, 125000, 625000]
][index]
samples = []

idx = 0

cummulated = 0
for scaling in tqdm(scalings, total=len(scalings)):
    while True:
        num_QAs = len(data[idx]['conversations']) // 2
        if cummulated + num_QAs <= scaling and idx < len(data) - 1:
            samples.append(data[idx])
            cummulated += num_QAs
            idx += 1
        else:
            if idx == len(data) - 1:
                1+1
            temp = deepcopy(data[idx])
            temp['conversations'] = temp['conversations'][:int(2*(scaling-cummulated))]
            temp = [temp]
            final = samples + temp if scaling-cummulated>0 else samples
            total_qas = sum(len(x['conversations']) // 2 for x in final)
            
            with open(path.replace('.json', f"_{total_qas}QAs.json"), "w") as f:
                json.dump(final, f)
                
            samples.append(data[idx])
            cummulated += num_QAs
            idx += 1
            break

        