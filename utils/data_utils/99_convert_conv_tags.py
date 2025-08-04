from misc import load_json, write_json, convert_conv_tags

old = 'llamafac'
new = 'llava'

# new, old = old, new

file = '/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/UIPro_processed/LlamaFactory_data/UIPro_v3_woWAE_1D4SeeClickWeb_1D4TextLoc+AITWAndConM2W-IntentGnd_4288k_merged_1726178_MAXTOK3200_llamafac.jsonl'

data = load_json(file)


new_data = convert_conv_tags(data, new, old)

new_file = file.replace(f"_{old}", f"_{new}")
write_json(new_data, new_file)

print(f"Write {len(new_data)} samples to {new_file}")