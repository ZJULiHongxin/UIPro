import os
from .clip_encoder import CLIPVisionTower
from .pix2struct_encoder import Pix2StructLargeVisionTower
from .davit import DaViTVisionTower
def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'davit' in vision_tower.lower():
        return DaViTVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'pix2struct' in vision_tower:
        return Pix2StructLargeVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
