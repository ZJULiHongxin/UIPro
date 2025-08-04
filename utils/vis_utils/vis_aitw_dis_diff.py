import matplotlib.pyplot as plt
import os, json
import numpy as np

# 20.8
data = json.load(open("/mnt/nvme0n1p1/hongxin_li/highres_autogui/utils/eval_utils/eval_results/androidcontrol/1108_SliME-Gemma-1p1-2B_5MBoxQAs+SimpUI5p3M+AutoGUI625k+6MobilePlanning380k/checkpoint-9048/11-12-16-33-18.json"))


# data = json.load(open("/mnt/nvme0n1p1/hongxin_li/AutoGUI/logs/uipro/1101_205231_test_uipro_func_pred_rec-motif-refexp-screenspot_rec-vwb/screenspot_rec_test.json")) # 16382

diffs = []

for x in data["logs"][0]['HL']:
    if x["GT_action"]["action_type"] != 'click': continue
    gt_target = np.array(x["GT_action"]["bbox"])
    gt_target = np.array([(gt_target[0]+gt_target[2])/2, (gt_target[1]+gt_target[3])/2])
    if x["action_pred"]["action_type"] != 'click': continue
    pred_target = np.array(x["action_pred"]["target"])/100
    
    diff = np.linalg.norm(pred_target-gt_target)
    diffs.append(diff)
    

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(diffs, bins=50, color='blue', alpha=0.7)
plt.title('Distribution of UI-Pro pretrained with 20.8M on AndroidControl-HL', fontsize=14)
plt.xlim([0,0.5])
plt.xlabel('Normalized Distance', fontsize=14)
plt.ylabel('Count', fontsize=14)

# Enlarge tick labels
plt.tick_params(axis='both', which='major', labelsize=12)
# Show the plot
plt.savefig("andcon_20p8M.png")
