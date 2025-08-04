import os, json, random, re
from tqdm import tqdm
SCALE = 100

DATASET_NAME = 'UIPro'

BOXPATTERN = r'\((\d+,\d+,\d+,\d+)\)'

PNTPATTERN =r'\((\d+),\s*(\d+)\)'

ROOT = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp"
sources = [
    "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/autogui_simpletasks_processed/autogui_1283623QAs_llava_merged_179300_MAXTOK650.json",
    "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/android_system_1021/android_system_samples_629523_merged_86088_MAXTOK650.json",
    "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/omniact_processed/omniact_6k_merged_693_MAXTOK650.json",
    "/mnt/vdb1/hongxin_li/GUICourse/guienv-train_llava_680k_merged_141778_MAXTOK650.json",
    "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/mobileviews_processed/mobileviews_1600k_merged_186081_MAXTOK650.json",
    "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/AndroidControl_processed/AndroidControl_1096015_merged_105932_MAXTOK650.json",
    "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/MultiUI_processed/MultiUI_gnd_ref_qa_5201k_merged_2268744_MAXTOK650.json",
    "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/WAE_processed/WAE_7558k_merged_985245_MAXTOK650.json",
    "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/WebUI_processed/WebUI_638k_merged_230995_MAXTOK650.json"
]

TASK_TYPES = ['textloc', 'ocr', 'titleidentification', 'icongnd', 'iconref', 'elemgnd', 'elemref', 'elemclass', 'funcpred_ground', 'funcgnd', 'funcref', 'funcpred_ref', 'intentgnd', 'webqa', 'embedqa', 'uicaption', 'widgetlist']

save_to_dir = os.path.join(ROOT, f'{DATASET_NAME}_processed')

save_all_to = os.path.join(save_to_dir, f'{DATASET_NAME}.json')
    
def balance_uipro_data():

    if not os.path.exists(save_all_to):
        merge = []
        for source in sources:
            print(f"Loading {source}")
            with open(source) as f:
                data = json.load(f)
                merge.extend(data)

        
        with open(save_all_to, "w") as f:
            json.dump(merge, f)
    else:
        print(f"Loading {save_all_to}")
        with open(save_all_to) as f:
            merge = json.load(f)

    qa_cnt = sum(len(x['conversations']) // 2 for x in merge)
    
    task_stats = {k:0 for k in TASK_TYPES}
    # for x in tqdm(merge, total=len(merge), desc='calc stats'):
    #     for task_id in x['task_ids']:
    #         for k in TASK_TYPES:
    #             if k in task_id:
    #                 task_stats[k] += 1
    #                 break

    task_remain_probs = [['intentgnd', 0.3], ['textloc', 0.4], ['ocr', 0.3]]

    for x in tqdm(merge, total=len(merge)):
        new_convs = []
        random.shuffle(task_remain_probs)
        
        for conv_i in range(0, len(x['conversations']), 2):
            remain = False
            task_id = x['task_ids'][conv_i//2]
            
            for task, prob in task_remain_probs:
                if task in task_id:
                    if random.random() <= prob:
                        remain = True
                    break # 1/5 的intentgnd样本
                    
            else: remain = True
        
            if remain:
                new_convs.append(x['conversations'][conv_i])
                new_convs.append(x['conversations'][conv_i+1])

        if len(new_convs) == 0:
            random_conv_i = random.randint(0, len(x['conversations'])//2-1)
            new_convs.extend(x['conversations'][random_conv_i*2:random_conv_i*2+2])
        
        x['conversations'] = new_convs
    
    qa_cnt_after = sum(len(x['conversations']) // 2 for x in merge)

    task_stats_after = {k:0 for k in TASK_TYPES}
    for x in tqdm(merge, total=len(merge), desc='calc stats'):
        for task_id in x['task_ids']:
            for k in TASK_TYPES:
                if k in task_id:
                    task_stats_after[k] += 1
                    break

    with open(save_all_to.replace(".json", f"_{qa_cnt_after//1000}k_info.json"), "w") as f:
        json.dump({'all': {'num_QAs': qa_cnt, 'stats': task_stats}, 'balanced': {'num_QAs': qa_cnt_after, 'stats': task_stats_after}}, f, indent=2)

    with open(save_all_to.replace(".json", f"_{qa_cnt_after//1000}k_sample.json"), "w") as f:
        json.dump(random.sample(merge, 512), f)

    with open(save_all_to.replace(".json", f"_{qa_cnt_after//1000}k.json"), "w") as f:
        json.dump(merge, f)

def check_box_pnt(sample):
    valid_convs, valid_task_ids, invalid_task_ids = [], [], []
    
    for conv_i in range(0, len(sample['conversations']), 2):
        task_id = sample['task_ids'][conv_i//2]
        
        is_valid = True
        if 'gnd' in task_id or 'loc' in task_id or 'ground' in task_id:
            gpt = sample['conversations'][conv_i+1]['value'][1:-1]
            coords = list(map(int, gpt.split(',')))

            if any(coord < 0 or coord >= SCALE for coord in coords):
                invalid_task_ids.append(task_id)
                is_valid = False
        elif 'ref' in task_id:
            user = sample['conversations'][conv_i]['value']
            coords = re.search(PNTPATTERN, user)
            
            if not coords:
                coords = re.search(BOXPATTERN, user)
            
            if coords:
                coords = list(map(int, coords.group(0).strip('()').split(',')))
                if any(coord < 0 or coord >= SCALE for coord in coords):
                    invalid_task_ids.append(task_id)
                    is_valid = False
        
        if is_valid:
            valid_convs.append(sample['conversations'][conv_i])
            valid_convs.append(sample['conversations'][conv_i+1])
            valid_task_ids.append(task_id)
    
    return valid_convs, valid_task_ids, invalid_task_ids

def clean_uipro_data():

    
    if not os.path.exists(save_all_to):
        merge = []
        for source in sources:
            print(f"Loading {source}")
            with open(source) as f:
                data = json.load(f)
                merge.extend(data)

        
        with open(save_all_to, "w") as f:
            json.dump(merge, f)
    else:
        print(f"Loading {save_all_to}")
        with open(save_all_to) as f:
            merge = json.load(f)
    
    qa_cnt_before = sum(len(x['conversations']) // 2 for x in merge)
    
    invalid_task_ids_aggr = []

    new_merge = []

    for sample in tqdm(merge, total=len(merge), desc='clean data'):
        valid_convs, valid_task_ids, invalid_task_ids = check_box_pnt(sample)
        invalid_task_ids_aggr.extend(invalid_task_ids)
        if len(invalid_task_ids):
            1+1
        
        sample['conversations'], sample['task_ids'] = valid_convs, valid_task_ids
        if len(sample['conversations']) == 0: continue
        new_merge.append(sample)

    qa_cnt_after = sum(len(x['conversations']) // 2 for x in new_merge)

    random.shuffle(new_merge)
    with open(save_all_to.replace(".json", f"_{qa_cnt_after//1000}k_sample.json"), "w") as f:
        json.dump(random.sample(new_merge,1024), f)

    with open(save_all_to.replace(".json", f"_{qa_cnt_after//1000}k.json"), "w") as f:
        json.dump(new_merge, f)

    with open(save_all_to.replace(".json", f"_{qa_cnt_after//1000}k_cleaning_info.json"), "w") as f:
        json.dump({'invalid_task_ids_aggr': invalid_task_ids_aggr}, f, indent=2)

clean_uipro_data()