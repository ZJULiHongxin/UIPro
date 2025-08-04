import json, os, cv2, numpy as np

index = 1
TASK = ['website', 'domain', 'task'][0]

ROOT = [
    "/mnt/nvme0n1p1/hongxin_li/UI_training_data/Mind2Web",
    "/mnt/nvme0n1p1/hongxin_li/UI_training_data/Mind2Web",
    "/data2/hongxin_li/UI_training_data/Mind2Web"][index]

mind2web_imgs_dir = os.path.join(ROOT, "mind2web_images")

mind2web_test = json.load(open(f'{ROOT}/mind2web_data_test_' + TASK + '.json', 'r'))

eval_res_file = "/mnt/nvme0n1p1/hongxin_li/highres_autogui/utils/eval_utils/eval_results/mind2web/1110_SliME-Gemma-1p1-2B_5MBoxQAs+AutoGUI625k+Mind2Web7p7k-WebLINX/checkpoint-2256/website-11-11-00-25-12.json"

eval_res = json.load(open(eval_res_file))

for episode in eval_res['log']:
    for sample in episode:
        action_ref = eval(sample['GT_action'])
        action_pred = eval(sample['response'][7:])
        
        target_pred = action_pred['target']
        normalized_gt_box = sample['GT_box']
        
        img = cv2.imread(os.path.join(mind2web_imgs_dir, sample['img_path']))
        
        H, W = img.shape[:2]
        
        target = [round(target_pred[0]/100*W), round(target_pred[1]/100*H)]
        gt_box = list(map(round, [normalized_gt_box[0]*W, normalized_gt_box[1]*H, normalized_gt_box[2]*W, normalized_gt_box[3]*H]))
        cv2.circle(img, center=target, radius=5, color=(0,0,250), thickness=2)
        cv2.rectangle(img, gt_box[:2], gt_box[2:], color=(0,250,0), thickness=2)
        
        print(sample['task'])
        print(sample['prompt'].split('history: ')[1].split('Now,')[0].strip())
        print(f"{action_ref}  <=> {action_pred}\n")
        
        cv2.imwrite("test.png", img)
        1+1