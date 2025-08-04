import json, os, cv2, numpy as np

index = 0

ROOT = [
    "/mnt/vdb1/hongxin_li/AndroidControl/images",
    "/data2/hongxin_li/UI_training_data/Mind2Web"][index]

eval_res_file = "/mnt/nvme0n1p1/hongxin_li/highres_autogui/utils/eval_utils/eval_results/androidcontrol/1108_SliME-Gemma-1p1-2B_5MBoxQAs+SimpUI5p3M+UIPro18p6M+AndroidControl124k/checkpoint-4845/05-15-16-14-14.json"

eval_res = json.load(open(eval_res_file))

for repeat_log in eval_res['logs']:
    for task_type, samples in repeat_log.items():
        for sample in samples:
            action_ref = sample['GT_action']
            action_pred = eval(sample['response'][7:])
            
            if action_ref['action_type'] != 'wait': continue
            
            img = cv2.imread(os.path.join(ROOT, sample['img_path']))
            
            H, W = img.shape[:2]

            if 'target' in action_pred:
                target_pred = action_pred['target']
                normalized_gt_box = sample['GT_action']['bbox']
                target = [round(target_pred[0]/100*W), round(target_pred[1]/100*H)]
                gt_box = list(map(round, [normalized_gt_box[0]*W, normalized_gt_box[1]*H, normalized_gt_box[2]*W, normalized_gt_box[3]*H]))
                cv2.circle(img, center=target, radius=5, color=(0,0,250), thickness=2)
                cv2.rectangle(img, gt_box[:2], gt_box[2:], color=(0,250,0), thickness=2)
            
            print(sample['task'])
            print(sample['prompt'].split('history: ')[1].split('Now,')[0].strip())
            print(f"{action_ref}  <=> {action_pred}\n")
            
            cv2.imwrite("test.png", img)
            1+1