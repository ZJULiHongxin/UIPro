# 筛选出各大benchmark所用的app上采集的样本
import json, os, random

selected_apps = [x[x.rfind('_')+1:] for x in json.load(open(os.path.join(os.path.dirname(__file__), 'selected_mobileview_apps.json')))]

ROOT = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/mobileviews_processed"
raw_mobileviews_sample_file = os.path.join(ROOT, 'mobileviews_149k.json')

raw_mobileviews_samples = json.load(open(raw_mobileviews_sample_file))

selected_samples = [x for x in raw_mobileviews_samples if x['package'] in selected_apps]

# resampling
resampled = selected_samples * (len(raw_mobileviews_samples) // len(selected_samples))
resampled.extend(random.sample(selected_samples, len(raw_mobileviews_samples) % len(selected_samples)))
random.shuffle(resampled)

with open(raw_mobileviews_sample_file.replace('.json', f'_selected{len(selected_samples)//1000}k_resample.json'), 'w') as f:
    json.dump(resampled, f)