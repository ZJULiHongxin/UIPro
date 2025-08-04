import os, json

from tqdm import tqdm

file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/UIPro_processed/LlamaFactory_data/UIPro_v3_woWAE_1D4SeeClickWeb_1D4TextLoc+AITWAndConM2W-IntentGnd_4295k_florence.jsonl"

with open(file, "r") as f:
    data = [json.loads(line) for line in f]

# AutoGUI
autogui, mobileviews, andcon, webui, multiui, seeclickweb, guienv, ricosca, motif, refexp, mind2web, aitw, omniact, others = [], [], [], [], [], [], [], [], [], [], [], [], [], []

for x in tqdm(data):
    img_path = x['images'][0].lower()
    if 'test_samples' in img_path or 'android_system' in img_path:
        autogui.append(x)
    elif 'mobileviews' in img_path:
        mobileviews.append(x)
    elif 'androidcontrol' in img_path:
        andcon.append(x)
    elif 'webui/' in img_path:
        webui.append(x)
    elif 'multiui/' in img_path:
        multiui.append(x)
    elif 'guienv/' in img_path:
        guienv.append(x)
    elif 'rico/' in img_path:
        ricosca.append(x)
    elif 'motif/' in img_path:
        motif.append(x)
    elif 'refexp/' in img_path:
        refexp.append(x)
    elif 'mind2web/' in img_path:
        mind2web.append(x)
    elif 'aitw/' in img_path:
        aitw.append(x)
    elif 'seeclick' in img_path:
        seeclickweb.append(x)
    elif 'omniact/' in img_path:
        omniact.append(x)
    else:
        others.append(x)
print(f"Total: {len(data)} = AutoGUI: {len(autogui)} + MobileViews: {len(mobileviews)} + AndroidControl: {len(andcon)} + WebUI: {len(webui)} + MultiUI: {len(multiui)} + GUIEnv: {len(guienv)} + Ricosca: {len(ricosca)} + Motif: {len(motif)} + RefExp: {len(refexp)} + Mind2Web: {len(mind2web)} + AITW: {len(aitw)} + SeeClickWeb: {len(seeclickweb)} + OmniAct: {len(omniact)} + Others: {len(others)}")

def judge_task(x):
    user_query = x['messages'][0]['content']
    if 'I want to ' in user_query:
        return 'intentgnd'
    elif 'Locate the text' in user_query:
        return 'textloc'
    elif 'ist all ' in user_query:
        return 'widlist'
    elif ' icon?' in user_query:
        return 'elemgnd'
    elif 'Locate the element according to its detailed functionality description' in user_query:
        return 'funcgnd'
    else:
        return 'elemgnd'

# AutoGUI task ratio
for task_portion, task_name in zip([autogui, mobileviews, andcon, webui, multiui, guienv, ricosca, motif, refexp, mind2web, aitw, seeclickweb, omniact, others], ['AutoGUI', 'MobileViews', 'AndroidControl', 'WebUI', 'MultiUI', 'GUIEnv', 'Ricosca', 'Motif', 'RefExp', 'Mind2Web', 'AITW', 'SeeClickWeb', 'OmniAct', 'Others']):
    intentgnd, textloc, widlist, elemgnd, funcgnd, others = [], [], [], [], [], []
    for x in task_portion:
        if judge_task(x) == 'intentgnd':
            intentgnd.append(x)
        elif judge_task(x) == 'textloc':
            textloc.append(x)
        elif judge_task(x) == 'widlist':
            widlist.append(x)
        elif judge_task(x) == 'elemgnd':
            elemgnd.append(x)
        elif judge_task(x) == 'funcgnd':
            funcgnd.append(x)
        else:
            others.append(x)

    print(f"{task_name} Total: {len(task_portion)} = IntentGnd: {len(intentgnd)} + TextLoc: {len(textloc)} + WidList: {len(widlist)} + ElemGnd: {len(elemgnd)} + FuncGnd: {len(funcgnd)} + Others: {len(others)}")

# total task tatio
intentgnd, textloc, widlist, elemgnd, funcgnd, others = [], [], [], [], [], []
for x in data:
    if judge_task(x) == 'intentgnd':
        intentgnd.append(x)
    elif judge_task(x) == 'textloc':
        textloc.append(x)
    elif judge_task(x) == 'widlist':
        widlist.append(x)
    elif judge_task(x) == 'elemgnd':
        elemgnd.append(x)
    elif judge_task(x) == 'funcgnd':
        funcgnd.append(x)
    else:
        others.append(x)

print(f"Total: {len(data)} = IntentGnd: {len(intentgnd)} + TextLoc: {len(textloc)} + WidList: {len(widlist)} + ElemGnd: {len(elemgnd)} + FuncGnd: {len(funcgnd)} + Others: {len(others)}")