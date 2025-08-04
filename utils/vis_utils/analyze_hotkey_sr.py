import os
import json
import numpy as np
import matplotlib.pyplot as plt

eval_result_path = "/mnt/nvme0n1p1/hongxin_li/highres_autogui/utils/eval_utils/eval_results/Omniact_desktop/0126_OS-Atlas-Base-7B_AblatedActSpace_s1000_6478/checkpoint-150_wActRef_relaxedGnd/01-26-17-35-27.json"

with open(eval_result_path, 'r') as f:
    eval_result = json.load(f)

hotkey_correct = {}

for x in eval_result['logs']:
    gt_action_type = x["GT_action"]["action_type"]

    if gt_action_type not in ('press_key', 'hotkey'):
        continue

    key = x["GT_action"]["key"]

    hotkey_correct.setdefault(key, []).append(x['metrics']['hotkey_match'])

print("\nHotkey Success Rates:")
print("-" * 40)
for key, matches in hotkey_correct.items():
    num_correct = sum(matches)
    total = len(matches)
    success_rate = (num_correct / total) * 100 if total > 0 else 0
    print(f"{key:<20} {success_rate:>6.1f}% ({num_correct}/{total})")
print("-" * 40)
