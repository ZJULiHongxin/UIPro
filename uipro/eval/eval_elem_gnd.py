import json, os
import transformers, torch
from tqdm import tqdm

from uipro.model.builder import load_pretrained_model

EAVL_DATASETS = {
    'funcpred': 'WebAgent/AutoGUI-v1-test',
    'screenspot': 'rootsautomation/ScreenSpot',
    'motif': 'HongxinLi/MOTIF-EVAL',
    'refexp': 1
}
def eval():
    model_name = ...
    
    