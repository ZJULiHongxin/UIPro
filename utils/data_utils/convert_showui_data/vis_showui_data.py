import os, json, cv2, re
from colorama import Fore, Style

index = 1
file = [
    "/data/hongxin_li/scaling_exp/ShowUI_processed//hf_train.json",
    "/mnt/shared-storage/groups/stepone_mm/lhx/ui_data/ShowUI/ShowUI-desktop/metadata/hf_train.json"
    ][-1]

data = json.load(open(file))

SCALE =[ 1000, 1][index]

def extract_integers(s):
    # Find all integers in the string using regex
    integers = re.findall(r'\d+', s)

    # Convert the string matches to integers
    return list(map(int, integers))

for item in data:
    # [{"img_url": "message/screen_1.png", "img_size": [3360, 2100], "element": [{"instruction": "message_ash", "bbox": [0.005654761904761905, 0.0880952380952381, 0.1836309523809524, 0.14952380952380953], "point": [0.0946, 0.1188]}, {"instruction": "A blue chat card with the word 'Sure!' in white text.", "bbox": [0.005654761904761905, 0.0880952380952381, 0.1836309523809524, 0.14952380952380953], "point": [0.0946, 0.1188]}, {"instruction": "The element is at the bottom of the conversation list.", "bbox": [0.005654761904761905, 0.0880952380952381, 0.1836309523809524, 0.14952380952380953], "point": [0.0946, 0.1188]}, {"instruction": "Read confirmation message from Ash.", "bbox": [0.005654761904761905, 0.0880952380952381, 0.1836309523809524, 0.14952380952380953], "point": [0.0946, 0.1188]}, {"instruction": "type
    img_url = item["img_url"]
    
    print(Fore.RED + f"Processing {img_url}" + Style.RESET_ALL)
    img_size = item["img_size"]
    element = item["element"]
    for ele in element:
        
        img = cv2.imread(img_url)
        H, W = img.shape[:2]
        instruction = ele["instruction"]
        
        print(Fore.GREEN + f"Processing {instruction}" + Style.RESET_ALL)
        
        if 'response' in ele: # <|box_start|>(817,428),(920,466)<|box_end|>
            print(Fore.YELLOW + f"Processing {ele['response']}" + Style.RESET_ALL)
            # extract the integers using Regex
            if 'list all' not in instruction:
                # Find all integers in the string using regex
                x1, y1, x2, y2 = extract_integers(ele['response'])
                x1, y1, x2, y2 = int(x1 / SCALE * W), int(y1 / SCALE * H), int(x2 / SCALE * W), int(y2 / SCALE * H)
                if x1 == x2 and y1 == y2:
                    cv2.circle(img, (x1, y1), 5, (0, 0, 255), -1)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                for line in ele['response'].split('\n'):
                    x1, y1, x2, y2 = extract_integers(line.split('<|box_start|>')[1])
                    x1, y1, x2, y2 = int(x1 / SCALE * W), int(y1 / SCALE * H), int(x2 / SCALE * W), int(y2 / SCALE * H)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        else:
            if (bbox := ele.get("bbox", None)) is not None:
                print(Fore.BLUE + f"Processing {bbox}" + Style.RESET_ALL)
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1 / SCALE * W), int(y1 / SCALE * H), int(x2 / SCALE * W), int(y2 / SCALE * H)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if (point := ele.get("point", None)) is not None:
                print(Fore.BLUE + f"Processing {point}" + Style.RESET_ALL)
                x, y = point
                x, y = int(x / SCALE * W), int(y / SCALE * H)
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                
        cv2.imwrite('test.png', img)
        print()