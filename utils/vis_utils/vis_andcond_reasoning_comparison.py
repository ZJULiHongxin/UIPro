import os, json, cv2
from colorama import Fore, Style
import gradio as gr
from typing import Dict, List

# androidcontrol_test_raw = json.load(open('/mnt/vdb1/hongxin_li/AndroidControl_test/AndroidControl-test_12685.json', 'r'))

# img_dict = {}
# for x in androidcontrol_test_raw:
#     task, step_imgpath = x['task'].strip(' .'), x['image']
#     step_id = '/'.join(step_imgpath.split('/')[-3:])
#     img_dict[f"{task}-{step_id}"] = {'image': x['image'], 'history': x['history']}
    
# reasoning_result_file = "utils/eval_utils/eval_results/androidcontrol/Qwen/Qwen2.5-VL-7B-Instruct/03-17-10-28-33.json"
# reasoning_eval_result = json.load(open(reasoning_result_file))

# img_dir = '/mnt/vdb1/hongxin_li/'

# logs=reasoning_eval_result["logs"]
# for eval_result_round in logs:
#     for sample in eval_result_round['H']:
#         if sample['metrics']['action_match']: continue

#         if sample['GT_action']['action_type'] != 'click': continue

#         if sample.get('wrong_format', False): continue

#         task, img_path, response, action_pred, gt_action = sample['task'], sample['img_path'], sample['response'], sample['action_pred'], sample['GT_action']
#         step_id = '/'.join(img_path.split('/')[-3:])
#         identifier = f"{task.strip(' .')}-{step_id}"
#         img_path = os.path.join(img_dir, img_dict[identifier]['image'])
#         history = img_dict[identifier]['history']
        
#         img = cv2.imread(img_path)
#         H, W = img.shape[:2]
#         if gt_action['action_type'] == 'click':
#             gt_x, gt_y = round(gt_action['x']), round(gt_action['y'])
#             cv2.circle(img, (gt_x, gt_y), radius=7, color=(0,0,255), thickness=2)
#             cv2.circle(img, (gt_x, gt_y), radius=2, color=(0,0,255), thickness=-1)
            
#             if 'target' in action_pred:
#                 pred_x, pred_y = round(action_pred['target'][0] / 1000 * W), round(action_pred['target'][1] / 1000 * H)
#                 cv2.circle(img, (pred_x, pred_y), radius=7, color=(0,255,255), thickness=2)

#         cv2.imwrite('test.png', img)
        
#         print(Fore.CYAN + f"Task: {task}\n" + Style.RESET_ALL + f"History: {history}\n\nGPT: {response}\nGT: {gt_action}")
#         1+1
            
def load_android_test_data(test_file_path: str) -> Dict[str, dict]:
    """Load and process the Android test data"""
    android_test_raw = json.load(open(test_file_path, 'r'))
    img_dict = {}
    for x in android_test_raw:
        task, step_imgpath = x['task'].strip(' .'), x['image']
        step_id = '/'.join(step_imgpath.split('/')[-3:])
        img_dict[f"{task}-{step_id}"] = {'image': x['image'], 'history': x['history']}
    return img_dict

def load_result_file(result_file: str) -> List[dict]:
    """Load and flatten the evaluation results"""
    eval_result = json.load(open(result_file))
    flat_samples = []
    for eval_round in eval_result["logs"]:
        for sample in eval_round['H']:
            if sample.get('wrong_format', False):
                continue
            flat_samples.append(sample)
    return flat_samples

def draw_action_on_image(img, action_pred, gt_action):
    """Draw predicted and ground truth actions on the image"""
    H, W = img.shape[:2]
    if gt_action['action_type'] == 'click':
        gt_x, gt_y = round(gt_action['x']), round(gt_action['y'])
        cv2.circle(img, (gt_x, gt_y), radius=7, color=(0,0,255), thickness=2)  # Red for ground truth
        cv2.circle(img, (gt_x, gt_y), radius=2, color=(0,0,255), thickness=-1)
        
        if 'target' in action_pred:
            pred_x, pred_y = round(action_pred['target'][0] / 1000 * W), round(action_pred['target'][1] / 1000 * H)
            cv2.circle(img, (pred_x, pred_y), radius=7, color=(0,255,255), thickness=2)  # Yellow for prediction
    return img

def create_comparison_interface(result_file1: str, result_file2: str, android_test_path: str, img_base_dir: str):
    """Create and launch the Gradio interface for result comparison"""
    img_dict = load_android_test_data(android_test_path)
    samples1 = load_result_file(result_file1)
    samples2 = load_result_file(result_file2)
    
    # Create lookup dictionary for second result file
    samples2_dict = {f"{s['task'].strip(' .')}-{'/'.join(s['img_path'].split('/')[-3:])}": s 
                     for s in samples2}

    def show_comparison(index: int):
        sample1 = samples1[index]
        task = sample1['task'].strip(' .')
        step_id = '/'.join(sample1['img_path'].split('/')[-3:])
        identifier = f"{task}-{step_id}"
        
        # Get corresponding sample from second result file
        sample2 = samples2_dict.get(identifier)
        if not sample2:
            return None, "No matching sample found", None, "N/A"
        
        # Load and process image
        img_path = os.path.join(img_base_dir, img_dict[identifier]['image'])
        img = cv2.imread(img_path)
        
        # Create two copies of the image for side-by-side comparison
        img1 = img.copy()
        img2 = img.copy()
        
        img1 = draw_action_on_image(img1, sample1['action_pred'], sample1['GT_action'])
        img2 = draw_action_on_image(img2, sample2['action_pred'], sample2['GT_action'])
        
        # Prepare display text with additional information
        history = img_dict[identifier]['history']
        correct1 = "✅" if sample1['metrics']['action_match'] else "❌"
        correct2 = "✅" if sample2['metrics']['action_match'] else "❌"
        
        text1 = (f"Prediction: {correct1}\n\n"
                f"Task: {task}\n\n"
                f"History: {history}\n\n"
                f"Ground Truth Action: {sample1['GT_action']}\n\n"
                f"Model Response:\n{sample1['response']}")
        
        text2 = (f"Prediction: {correct2}\n\n"
                f"Task: {task}\n\n"
                f"History: {history}\n\n"
                f"Ground Truth Action: {sample2['GT_action']}\n\n"
                f"Model Response:\n{sample2['response']}")
        
        return img1, text1, img2, text2

    with gr.Blocks() as interface:
        gr.Markdown("# Model Response Comparison")
        with gr.Row():
            index_slider = gr.Slider(minimum=0, maximum=len(samples1)-1, step=1, label="Sample Index")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"### Model 1 ({os.path.basename(result_file1)})")
                image1 = gr.Image(label="Image with Actions")
                text1 = gr.Textbox(label="Details", lines=15)
            
            with gr.Column():
                gr.Markdown(f"### Model 2 ({os.path.basename(result_file2)})")
                image2 = gr.Image(label="Image with Actions")
                text2 = gr.Textbox(label="Details", lines=15)
        
        index_slider.change(
            show_comparison,
            inputs=[index_slider],
            outputs=[image1, text1, image2, text2]
        )
    
    return interface

if __name__ == "__main__":
    # Configuration
    RESULT_FILE1 = "utils/eval_utils/eval_results/androidcontrol/Qwen/Qwen2.5-VL-7B-Instruct/03-17-10-28-33.json"
    RESULT_FILE2 = "utils/eval_utils/eval_results/androidcontrol/Qwen/Qwen2.5-VL-72B-Instruct/03-19-01-57-45.json"  # Replace with your second result file
    ANDROID_TEST_PATH = "/mnt/vdb1/hongxin_li/AndroidControl_test/AndroidControl-test_12685.json"
    IMG_BASE_DIR = "/mnt/vdb1/hongxin_li/"
    
    interface = create_comparison_interface(RESULT_FILE1, RESULT_FILE2, ANDROID_TEST_PATH, IMG_BASE_DIR)
    interface.launch(share=True)
        
        
        