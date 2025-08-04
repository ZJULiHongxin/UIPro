import os, json, cv2
from colorama import Fore, Style

androidcontrol_test_raw = json.load(open('/mnt/vdb1/hongxin_li/AndroidControl_test/AndroidControl-test_12685.json', 'r'))

img_dict = {}
for x in androidcontrol_test_raw:
    task, step_imgpath = x['task'].strip(' .'), x['image']
    step_id = '/'.join(step_imgpath.split('/')[-3:])
    img_dict[f"{task}-{step_id}"] = {'image': x['image'], 'history': x['history']}
    
reasoning_result_file = "utils/eval_utils/eval_results/androidcontrol/Qwen/Qwen2.5-VL-7B-Instruct/03-17-10-28-33.json"
reasoning_eval_result = json.load(open(reasoning_result_file))

img_dir = '/mnt/vdb1/hongxin_li/'

logs=reasoning_eval_result["logs"]
for eval_result_round in logs:
    for sample in eval_result_round['H']:
        if sample['metrics']['action_match']: continue

        if sample['GT_action']['action_type'] != 'click': continue

        if sample.get('wrong_format', False): continue

        task, img_path, response, action_pred, gt_action = sample['task'], sample['img_path'], sample['response'], sample['action_pred'], sample['GT_action']
        step_id = '/'.join(img_path.split('/')[-3:])
        identifier = f"{task.strip(' .')}-{step_id}"
        img_path = os.path.join(img_dir, img_dict[identifier]['image'])
        history = img_dict[identifier]['history']
        
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        if gt_action['action_type'] == 'click':
            gt_x, gt_y = round(gt_action['x']), round(gt_action['y'])
            cv2.circle(img, (gt_x, gt_y), radius=7, color=(0,0,255), thickness=2)
            cv2.circle(img, (gt_x, gt_y), radius=2, color=(0,0,255), thickness=-1)
            
            if 'target' in action_pred:
                pred_x, pred_y = round(action_pred['target'][0] / 1000 * W), round(action_pred['target'][1] / 1000 * H)
                cv2.circle(img, (pred_x, pred_y), radius=7, color=(0,255,255), thickness=2)

        cv2.imwrite('test.png', img)
        
        print(Fore.CYAN + f"Task: {task}\n" + Style.RESET_ALL + f"History: {history}\n\nGPT: {response}\nGT: {gt_action}")
        1+1
            
        
        
        