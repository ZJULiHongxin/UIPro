"""
GUICourse Dataset Processing Pipeline

This module processes GUI interaction datasets from multiple sources (GUIEnv, GUIChat, GUIAct)
and converts them into standardized formats for training multimodal AI models on GUI understanding
and interaction tasks.

The pipeline supports three main dataset types:
- GUIEnv: Text-to-location and OCR tasks
- GUIChat: Conversational GUI interactions with bounding boxes
- GUIAct: Action planning and intent grounding tasks

Author: Seasoned Engineering Team
Version: 2.0.0
"""

import ast
import cv2
import glob
import json
import os
import random
import re
from collections import defaultdict
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
from rapidfuzz import fuzz
from tqdm import tqdm

from utils.data_utils.task_prompt_lib import *
from utils.data_utils.misc import (
    is_pure_color, 
    is_valid_string, 
    decode_img_base64, 
    keep_unique_actions
)

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

class DatasetConfig:
    """Configuration class containing all dataset processing parameters."""
    
    # Dataset identification
    DATASET_NAME = 'GUICourse'
    
    # Subtask
    SUBTASK = ['GUIAct'] # ['GUIEnv', 'GUIChat', 'GUIAct']
    
    # Random seed for reproducibility
    RANDOM_SEED = 666
    
    # Data split configuration
    SPLIT_OPTIONS = ['train', 'test']
    CURRENT_SPLIT = SPLIT_OPTIONS[0]  # 'train'
    
    # Device type configuration
    DEVICE_TYPES = ['web', 'smartphone']
    CURRENT_DEVICE_TYPE = DEVICE_TYPES[1]  # 'smartphone'
    
    # Path configurations
    DATA_ROOT = "/mnt/jfs/copilot/lhx/ui_data/GUICourse"
    SAVE_DIR = f"/data/hongxin_li/scaling_exp/{DATASET_NAME}_processed"
    
    # Processing flags
    DEBUG_MODE = False
    ENABLE_TEXTLOC = False
    ENABLE_OCR = False
    ENABLE_INTENT_GND = False
    USE_ACTION_PROMPT = False
    SKIP_VALIDATION = False
    
    # Output formatting
    OUTPUT_TAG = ''  # e.g., 'Please output its center coordinates.'
    LOCATION_FORMAT = None  # e.g., 'action_json'
    
    # Coordinate scaling and box probability
    COORDINATE_SCALE = 1000
    BOUNDING_BOX_PROBABILITY = 0.0
    
    # Action configuration
    USE_ACTION_REFEXP = True
    MAX_PREVIOUS_ACTIONS = 3  # Maximum number of previous actions to include in history
    
    # Validation thresholds
    MIN_ELEMENT_SIZE_RATIO = 0.005  # Minimum element size relative to image dimensions
    MIN_OCR_SIMILARITY_THRESHOLD = 22  # Minimum similarity for OCR validation

# Task mapping for different operation types
TASK_TYPE_MAPPING = {
    'text2bbox': 'textloc',
    'bbox2text': 'ocr'
}

# Invalid element categories for tracking data quality
INVALID_ELEMENT_CATEGORIES = {
    'TOO_SMALL_ELEMENT': 'too_small',
    'INVALID_ELEM_BOX': 'invalid_box',
    'INVALID_ELEM_CONTENT': 'invalid_content',
    'BLANK_ELEM': 'blank',
    'EMPTY_ELEM_TEXT': 'empty_text',
    'OVERLY_LENGTHY_ELEM_TEXT': 'lengthy_text',
    'DUPLICATE_ELEMENT': 'duplicate',
    'INCORRECT_TEXT_ANNO': 'incorrect_annotation'
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_environment() -> None:
    """Initialize the processing environment with proper configuration."""
    random.seed(DatasetConfig.RANDOM_SEED)
    os.makedirs(DatasetConfig.SAVE_DIR, exist_ok=True)

def load_invalid_elements(filepath: str) -> Dict[str, set]:
    """
    Load previously identified invalid elements from file.
    
    Args:
        filepath: Path to the invalid elements record file
        
    Returns:
        Dictionary mapping category names to sets of invalid element identifiers
    """
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
        return {k: set(v) for k, v in loaded_data.items()}
    
    # Initialize empty sets for each category
    return {category: set() for category in INVALID_ELEMENT_CATEGORIES.values()}

def save_invalid_elements(invalid_elements: Dict[str, set], filepath: str) -> None:
    """
    Save invalid elements record to file.
    
    Args:
        invalid_elements: Dictionary of invalid element categories and their IDs
        filepath: Path where to save the record
    """
    serializable_data = {k: list(v) for k, v in invalid_elements.items()}
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=2)

def validate_element_box(box: List[float], image_size: Tuple[int, int]) -> bool:
    """
    Validate if an element bounding box is geometrically valid.
    
    Args:
        box: Bounding box coordinates [x1, y1, x2, y2]
        image_size: Image dimensions (width, height)
        
    Returns:
        True if the box is valid, False otherwise
    """
    x1, y1, x2, y2 = box
    width, height = image_size
    
    # Check if coordinates are within image bounds
    if not (0 <= x1 <= width and 0 <= x2 <= width and 
            0 <= y1 <= height and 0 <= y2 <= height):
        return False
    
    # Check if box has positive area
    if not (x1 < x2 and y1 < y2):
        return False
    
    # Check if box is not too small
    width_ratio = abs(x1 - x2) / width
    height_ratio = abs(y1 - y2) / height
    
    return (width_ratio > DatasetConfig.MIN_ELEMENT_SIZE_RATIO and 
            height_ratio > DatasetConfig.MIN_ELEMENT_SIZE_RATIO)

def normalize_coordinates(box: List[float], image_size: Tuple[int, int], 
                         scale: int = None) -> List[int]:
    """
    Normalize coordinates to a specified scale.
    
    Args:
        box: Original bounding box coordinates [x1, y1, x2, y2]
        image_size: Image dimensions (width, height)
        scale: Target scale (defaults to DatasetConfig.COORDINATE_SCALE)
        
    Returns:
        Normalized coordinates as list of integers
    """
    if scale is None:
        scale = DatasetConfig.COORDINATE_SCALE
    
    width, height = image_size
    x1, y1, x2, y2 = box
    
    return [
        round(x1 / width * scale),
        round(y1 / height * scale),
        round(x2 / width * scale),
        round(y2 / height * scale)
    ]

def validate_ocr_text(image: np.ndarray, box: List[int], expected_text: str) -> bool:
    """
    Validate OCR text annotation using Tesseract.
    
    Args:
        image: Source image as numpy array
        box: Bounding box coordinates [x1, y1, x2, y2]
        expected_text: Expected text content
        
    Returns:
        True if OCR validation passes, False otherwise
    """
    if not is_valid_string(expected_text):
        return True  # Skip validation for non-ASCII strings
    
    x1, y1, x2, y2 = box
    cropped_region = image[y1:y2, x1:x2]
    
    try:
        gray_region = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
        ocr_result = pytesseract.image_to_string(gray_region).strip()
        similarity = fuzz.ratio(ocr_result.lower(), expected_text.lower())
        
        return similarity >= DatasetConfig.MIN_OCR_SIMILARITY_THRESHOLD
    except Exception:
        return False  # Skip validation on OCR errors

def discretize_distance(distance: float) -> str:
    """
    Discretize continuous distance values into categorical bins.
    
    Args:
        distance: Continuous distance value
        
    Returns:
        Discretized distance category as string
    """
    if distance < 0.1:
        return "short"
    elif distance < 0.3:
        return "medium"
    else:
        return "long"

# =============================================================================
# DATASET PROCESSING CLASSES
# =============================================================================

class GUIEnvProcessor:
    """Processor for GUIEnv dataset - handles text-to-location and OCR tasks."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.invalid_elements = {}
        self.unique_elements = {}
        self.processed_samples = []
        self.statistics = {
            'textloc_count': 0,
            'ocr_count': 0,
            'total_iterations': 0
        }
    
    def process_dataset(self) -> List[Dict[str, Any]]:
        """
        Process the complete GUIEnv dataset.
        
        Returns:
            List of processed samples ready for training
        """
        if not self.config.ENABLE_TEXTLOC and not self.config.ENABLE_OCR:
            print("GUIEnv processing disabled (both TEXTLOC and OCR are False)")
            return []
        
        dataset_path = os.path.join(self.config.DATA_ROOT, "GUIEnv")
        parquet_files = glob.glob(os.path.join(dataset_path, "*.parquet"))
        
        # Initialize invalid elements tracking
        invalid_record_file = os.path.join(self.config.SAVE_DIR, 'invalid_elem_record.json')
        self.invalid_elements = load_invalid_elements(invalid_record_file)
        
        print(f"Processing {len(parquet_files)} GUIEnv parquet files...")
        
        for parquet_file in parquet_files:
            if self.config.CURRENT_SPLIT not in parquet_file:
                continue
            
            print(f"Processing {parquet_file}")
            self._process_parquet_file(parquet_file, dataset_path)
        
        # Save final statistics and invalid elements
        save_invalid_elements(self.invalid_elements, invalid_record_file)
        self._generate_statistics_report()
        
        return self.processed_samples
    
    def _process_parquet_file(self, parquet_file: str, dataset_path: str) -> None:
        """Process a single parquet file from GUIEnv dataset."""
        # Load data and metadata
        gui_data = pd.read_parquet(parquet_file)
        metadata_file = parquet_file.replace("images.parquet", "data.json")
        
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        # Group samples by image ID
        samples_by_image = defaultdict(list)
        for sample in metadata:
            samples_by_image[sample['image_id']].append(sample)
        
        # Ensure images are saved
        img_save_path = os.path.join(dataset_path, "imgs")
        os.makedirs(img_save_path, exist_ok=True)
        self._save_images(gui_data, img_save_path)
        
        # Process each image group
        for group_idx, (img_id, samples) in tqdm(
            enumerate(samples_by_image.items()), 
            total=len(samples_by_image), 
            desc='Processing GUIEnv samples...'
        ):
            self._process_image_group(img_id, samples, img_save_path, group_idx, len(samples_by_image))
    
    def _save_images(self, gui_data: pd.DataFrame, img_save_path: str) -> None:
        """Save base64 encoded images to disk."""
        for _, row in tqdm(gui_data.iterrows(), total=len(gui_data), desc='Saving images...'):
            img_id = row.name
            img_path = os.path.join(img_save_path, f"{img_id}.png")
            
            if os.path.exists(img_path):
                continue
            
            try:
                image_data = decode_img_base64(row.iloc[0])
                image = Image.open(BytesIO(image_data))
                image.save(img_path)
            except Exception as e:
                print(f"Failed to save image {img_id}: {e}")
    
    def _process_image_group(self, img_id: str, samples: List[Dict], img_save_path: str,
                           group_idx: int, total_groups: int) -> None:
        """Process all samples for a single image."""
        img_path = os.path.join(img_save_path, f"{img_id}.png")
        img = None
        used_instructions = []
        
        if img_id not in self.unique_elements:
            self.unique_elements[img_id] = []
        
        for sample in samples:
            processed_sample = self._process_single_sample(
                sample, img_path, img, used_instructions
            )
            
            if processed_sample is not None:
                self.processed_samples.append(processed_sample)
                if img is None:  # Load image only when needed
                    img = cv2.imread(img_path)
        
        # Periodic saving of invalid elements record
        if group_idx > 0 and (group_idx % 10000 == 0 or group_idx == total_groups - 1):
            invalid_record_file = os.path.join(self.config.SAVE_DIR, 'invalid_elem_record.json')
            save_invalid_elements(self.invalid_elements, invalid_record_file)
    
    def _process_single_sample(self, sample: Dict, img_path: str, img: Optional[np.ndarray],
                             used_instructions: List[str]) -> Optional[Dict[str, Any]]:
        """Process a single sample from the dataset."""
        # Parse sample information
        uid_parts = sample['uid'].split("_")
        if len(uid_parts) < 6:
            return None
        
        task_type = TASK_TYPE_MAPPING.get(uid_parts[4])
        if not task_type:
            return None
        
        # Check if task type is enabled
        if task_type == 'textloc' and not self.config.ENABLE_TEXTLOC:
            return None
        if task_type == 'ocr' and not self.config.ENABLE_OCR:
            return None
        
        self.statistics['total_iterations'] += 1
        sample_id = sample['uid']
        
        # Check if sample is already marked as invalid
        if self._is_sample_invalid(sample_id):
            return None
        
        # Extract task-specific information
        if task_type == 'textloc':
            instruction = sample['question'].strip()
            boxes = sample['answer']['absolute']
            
            if len(boxes) != 1:  # Only process single-target tasks
                return None
            
            box = list(map(int, boxes[0][5:-6].split(',')))
            
            # Check for duplicate instructions
            if not self.config.SKIP_VALIDATION and instruction in used_instructions:
                self.invalid_elements[INVALID_ELEMENT_CATEGORIES['DUPLICATE_ELEMENT']].add(sample_id)
                return None
            
            used_instructions.append(instruction)
            
        else:  # OCR task
            instruction = sample['answer'].strip()
            box = list(map(int, sample['question']['absolute'][5:-6].split(',')))
        
        # Validate sample
        if not self._validate_sample(sample, sample_id, instruction, box, img_path):
            return None
        
        # Generate processed sample
        return self._create_processed_sample(sample, task_type, instruction, box, img_path)
    
    def _is_sample_invalid(self, sample_id: str) -> bool:
        """Check if a sample is marked as invalid."""
        return any(sample_id in invalid_set for invalid_set in self.invalid_elements.values())
    
    def _validate_sample(self, sample: Dict, sample_id: str, instruction: str, 
                        box: List[int], img_path: str) -> bool:
        """Validate a sample against various quality criteria."""
        if self.config.SKIP_VALIDATION:
            return True
        
        # Check for empty instruction
        if len(instruction) == 0:
            self.invalid_elements[INVALID_ELEMENT_CATEGORIES['EMPTY_ELEM_TEXT']].add(sample_id)
            return False
        
        # Validate bounding box geometry
        image_size = (sample['image_size']['width'], sample['image_size']['height'])
        if not validate_element_box(box, image_size):
            self.invalid_elements[INVALID_ELEMENT_CATEGORIES['INVALID_ELEM_BOX']].add(sample_id)
            return False
        
        # Load image for visual validation
        img = cv2.imread(img_path)
        if img is None:
            return False
        
        # Check for blank/pure color regions
        if is_pure_color(img, box):
            self.invalid_elements[INVALID_ELEMENT_CATEGORIES['BLANK_ELEM']].add(sample_id)
            return False
        
        # Validate OCR annotation if applicable
        if not validate_ocr_text(img, box, instruction):
            self.invalid_elements[INVALID_ELEMENT_CATEGORIES['INCORRECT_TEXT_ANNO']].add(sample_id)
            return False
        
        return True
    
    def _create_processed_sample(self, sample: Dict, task_type: str, instruction: str,
                               box: List[int], img_path: str) -> Dict[str, Any]:
        """Create a processed sample in the target format."""
        # Extract image dimensions and normalize coordinates
        width, height = sample['image_size']['width'], sample['image_size']['height']
        x1, y1, x2, y2 = box
        
        normalized_box = normalize_coordinates(box, (width, height))
        center_x = round((x1 + x2) / 2 / width * self.config.COORDINATE_SCALE)
        center_y = round((y1 + y2) / 2 / height * self.config.COORDINATE_SCALE)
        
        # Determine location format (box vs point)
        if (x1 != x2 and y1 != y2 and 
            random.random() < self.config.BOUNDING_BOX_PROBABILITY):
            location = '(' + ','.join(map(str, normalized_box)) + ')'
            with_box = True
        else:
            location = f'({center_x},{center_y})'
            with_box = False
        
        # Generate relative image path
        short_img_path = img_path[img_path.find(self.config.DATASET_NAME):]
        
        # Create task-specific sample
        if task_type == 'textloc':
            if self.config.USE_ACTION_PROMPT:
                action = CLICK_TEMPLATE.format(target_x=center_x, target_y=center_y)
                processed_sample = make_actionplanning_sample(
                    task_id=f'autogui_{self.config.DATASET_NAME}_textloc_{len(self.processed_samples)}',
                    global_task=instruction,
                    gt_action=action,
                    history='None',
                    prompt_format_type='aguvis'
                )
            else:
                processed_sample = make_textloc_sample(
                    task_id=f"autogui_GUIEnv_{task_type}_{sample['uid'].split('_')[-1]}",
                    text=instruction,
                    loc=location,
                    output_tag=self.config.OUTPUT_TAG,
                    foramt=self.config.LOCATION_FORMAT
                )
            
            self.statistics['textloc_count'] += 1
        else:  # OCR
            processed_sample = make_ocr_sample(
                task_id=f"autogui_GUIEnv_{task_type}_{sample['uid'].split('_')[-1]}",
                text=instruction,
                loc=location,
                with_box=with_box
            )
            self.statistics['ocr_count'] += 1
        
        # Add metadata
        processed_sample.update({
            "image": short_img_path,
            "task_attr": instruction if task_type == 'textloc' else location,
            "wxh": f"{width}x{height}",
            "unnormalized_box": box
        })
        
        return processed_sample
    
    def _generate_statistics_report(self) -> None:
        """Generate and print processing statistics."""
        num_invalid = sum(len(v) for v in self.invalid_elements.values())
        num_unique_elements = sum(len(v) for v in self.unique_elements.values())
        num_valid_images = len([k for k, v in self.unique_elements.items() if len(v)])
        
        report = f"""
GUIEnv Processing Report:
------------------------
Total samples: {len(self.processed_samples)}
Unique elements: {num_unique_elements}
Valid unique images: {num_valid_images}
Total unique images: {len(self.unique_elements)}
Invalid element ratio: {num_invalid} / {self.statistics['total_iterations']} = {num_invalid/max(1, self.statistics['total_iterations']):.2f}
TextLoc count: {self.statistics['textloc_count']}
OCR count: {self.statistics['ocr_count']}
        """
        print(report)

class GUIChatProcessor:
    """Processor for GUIChat dataset - handles conversational GUI interactions."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.processed_samples = []
    
    def process_dataset(self) -> List[Dict[str, Any]]:
        """Process the complete GUIChat dataset."""
        dataset_path = os.path.join(self.config.DATA_ROOT, "GUIChat")
        parquet_files = glob.glob(os.path.join(dataset_path, "*.parquet"))
        
        print(f"Processing {len(parquet_files)} GUIChat parquet files...")
        
        for parquet_file in parquet_files:
            print(f"Processing {parquet_file}")
            self._process_parquet_file(parquet_file, dataset_path)
        
        print(f"Generated {len(self.processed_samples)} GUIChat samples")
        return self.processed_samples
    
    def _process_parquet_file(self, parquet_file: str, dataset_path: str) -> None:
        """Process a single parquet file from GUIChat dataset."""
        gui_data = pd.read_parquet(parquet_file)
        metadata_file = parquet_file.replace("images.parquet", "data.json")
        
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        # Group samples by image ID
        samples_by_image = defaultdict(list)
        for sample in metadata:
            samples_by_image[sample['image_id']].append(sample)
        
        # Setup image save path
        img_save_path = os.path.join(dataset_path, "imgs")
        os.makedirs(img_save_path, exist_ok=True)
        
        # Process each image group
        for img_id, samples in tqdm(
            samples_by_image.items(), 
            total=len(samples_by_image), 
            desc='Processing GUIChat samples...'
        ):
            self._process_chat_samples(img_id, samples, img_save_path)
    
    def _process_chat_samples(self, img_id: str, samples: List[Dict], img_save_path: str) -> None:
        """Process chat samples for a single image."""
        img_path = os.path.join(img_save_path, f"{img_id}.png")
        short_img_path = img_path[img_path.find(self.config.DATASET_NAME):]
        
        for sample in samples:
            processed_sample = self._create_chat_sample(sample, img_path, short_img_path)
            if processed_sample:
                self.processed_samples.append(processed_sample)
    
    def _create_chat_sample(self, sample: Dict, img_path: str, short_img_path: str) -> Optional[Dict[str, Any]]:
        """Create a processed chat sample with normalized bounding boxes."""
        try:
            # Load image to get dimensions
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            height, width = img.shape[:2]
            
            new_conversations = []
            unnormalized_boxes = []
            
            for turn in sample['text']:
                if turn['from'] == 'human':
                    # Clean user query and add bbox requirement
                    user_query = re.sub(r'<image>.*?</image>', '', turn['value']).strip()
                    user_query += ' (with bbox) (with <box></box> tags)'
                    new_conversations.append({'from': 'human', 'value': user_query})
                
                elif turn['from'] == 'gpt':
                    # Process assistant response with bounding boxes
                    processed_response, boxes = self._process_gpt_response(
                        turn['value'], width, height
                    )
                    unnormalized_boxes.extend(boxes)
                    new_conversations.append({'from': 'gpt', 'value': processed_response})
            
            return {
                'id': f"autogui_GUIChat_webqa_{len(self.processed_samples)}",
                'conversations': new_conversations,
                'image': short_img_path,
                'unnormalized_boxes': unnormalized_boxes,
                'wxh': f"{width}x{height}"
            }
            
        except Exception as e:
            print(f"Error processing chat sample {sample.get('uid', 'unknown')}: {e}")
            return None
    
    def _process_gpt_response(self, response: str, width: int, height: int) -> Tuple[str, List[List[int]]]:
        """Process GPT response to normalize bounding box coordinates."""
        # Find all bounding box annotations
        box_matches = list(re.finditer(r'<box>(.*?)</box>', response))
        
        if not box_matches:
            return response, []
        
        new_response = ''
        last_end = 0
        unnormalized_boxes = []
        
        for match in box_matches:
            # Extract coordinates
            box_coords = match.group(1)
            try:
                x1, y1, x2, y2 = map(int, box_coords.split())
                
                # Normalize coordinates to target scale
                normalized_box = [
                    max(0, min(self.config.COORDINATE_SCALE - 1, 
                              round(x1 / 1000 * self.config.COORDINATE_SCALE))),
                    max(0, min(self.config.COORDINATE_SCALE - 1, 
                              round(y1 / 1000 * self.config.COORDINATE_SCALE))),
                    max(0, min(self.config.COORDINATE_SCALE - 1, 
                              round(x2 / 1000 * self.config.COORDINATE_SCALE))),
                    max(0, min(self.config.COORDINATE_SCALE - 1, 
                              round(y2 / 1000 * self.config.COORDINATE_SCALE)))
                ]
                
                # Convert to actual pixel coordinates for storage
                actual_coords = [
                    round(x1 / 1000 * width),
                    round(y1 / 1000 * height),
                    round(x2 / 1000 * width),
                    round(y2 / 1000 * height)
                ]
                unnormalized_boxes.append(actual_coords)
                
                # Build new response with normalized coordinates
                new_box_str = f"<box>({','.join(map(str, normalized_box))})</box>"
                new_response += response[last_end:match.start()] + new_box_str
                last_end = match.end()
                
            except ValueError:
                # Invalid box format, keep original
                new_response += response[last_end:match.end()]
                last_end = match.end()
        
        new_response += response[last_end:]
        return new_response, unnormalized_boxes

class GUIActProcessor:
    """Processor for GUIAct dataset - handles action planning and intent grounding."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.processed_samples = []
        self.statistics = {
            'action_planning_count': 0,
            'intent_grounding_count': 0
        }
    
    def process_dataset(self) -> List[Dict[str, Any]]:
        """Process the complete GUIAct dataset."""
        dataset_path = os.path.join(self.config.DATA_ROOT, "GUIAct")
        parquet_files = glob.glob(os.path.join(dataset_path, "*.parquet"))
        
        print(f"Processing {len(parquet_files)} GUIAct parquet files...")
        
        for parquet_file in parquet_files:
            if (self.config.CURRENT_SPLIT not in parquet_file or 
                self.config.CURRENT_DEVICE_TYPE not in parquet_file):
                continue
            
            print(f"Processing {parquet_file}")
            self._process_parquet_file(parquet_file, dataset_path)
        
        self._generate_statistics_report()
        return self.processed_samples
    
    def _process_parquet_file(self, parquet_file: str, dataset_path: str) -> None:
        """Process a single parquet file from GUIAct dataset."""
        gui_data = pd.read_parquet(parquet_file)
        metadata_file = parquet_file.replace("images.parquet", "data.json")
        
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        # Group trajectories by episode ID
        trajectories = defaultdict(list)
        for sample in metadata:
            sample = self._parse_sample_metadata(sample, parquet_file)
            if sample and self._should_process_sample(sample, parquet_file):
                trajectories[sample['episode_id']].append(sample)
        
        # Sort each trajectory by image_id
        for trajectory in trajectories.values():
            trajectory.sort(key=lambda x: x['image_id'])
        
        # Setup image save path
        img_save_path = os.path.join(dataset_path, "imgs")
        os.makedirs(img_save_path, exist_ok=True)
        
        # Process trajectories
        self._process_trajectories(trajectories, img_save_path)
    
    def _parse_sample_metadata(self, sample: Dict, parquet_file: str) -> Optional[Dict[str, Any]]:
        """Parse and enrich sample metadata based on dataset subset."""
        try:
            uid_parts = sample['uid'].split("_")
            
            if 'web-single' in parquet_file:
                if len(uid_parts) < 5:
                    return None
                _, _, img_id, task_type, idx = uid_parts
                sample.update({
                    'task_tag': 'web-single',
                    'episode_id': img_id,
                    'sample_id': img_id,
                    'step_id': 0
                })
            elif 'web-multi' in parquet_file:
                if len(uid_parts) < 5:
                    return None
                _, _, ep_id, _, step_id = uid_parts
                sample.update({
                    'task_tag': 'web-multi',
                    'episode_id': ep_id,
                    'sample_id': f"{ep_id}-{step_id}",
                    'step_id': int(step_id)
                })
            elif 'smartphone' in parquet_file:
                if len(uid_parts) < 5:
                    return None
                _, _, ep_id, _, step_id = uid_parts
                sample.update({
                    'task_tag': 'smartphone',
                    'episode_id': ep_id,
                    'sample_id': f"{ep_id}-{step_id}",
                    'step_id': int(step_id)
                })
            else:
                return None
            
            return sample
            
        except (ValueError, IndexError) as e:
            print(f"Error parsing sample metadata: {e}")
            return None
    
    def _should_process_sample(self, sample: Dict, parquet_file: str) -> bool:
        """Determine if a sample should be processed based on filtering criteria."""
        # Skip samples with multiple actions for web-single
        if 'web-single' in parquet_file and len(sample['actions_label']) > 1:
            return False
        
        return True
    
    def _process_trajectories(self, trajectories: Dict[str, List[Dict]], img_save_path: str) -> None:
        """Process all trajectories in the dataset."""
        for item_idx, (episode_id, trajectory) in tqdm(
            enumerate(trajectories.items()), 
            total=len(trajectories), 
            desc=f'Processing {len(trajectories)} trajectories...'
        ):
            if self.config.DEBUG_MODE and item_idx % 100 != 0:
                continue
            
            self._process_single_trajectory(episode_id, trajectory, img_save_path)
    
    def _process_single_trajectory(self, episode_id: str, trajectory: List[Dict], img_save_path: str) -> None:
        """Process a single trajectory containing multiple steps."""
        step_instructions = []
        
        for step_idx, sample in enumerate(trajectory):
            processed_sample = self._process_action_step(
                sample, step_idx, step_instructions, img_save_path
            )
            
            if processed_sample:
                self.processed_samples.append(processed_sample)
                self.statistics['action_planning_count'] += 1
    
    def _process_action_step(self, sample: Dict, step_idx: int, step_instructions: List[str],
                           img_save_path: str) -> Optional[Dict[str, Any]]:
        """Process a single action step within a trajectory."""
        try:
            # Get action information
            action_info = self._extract_action_info(sample)
            if not action_info:
                return None
            
            action_name = action_info['name'].lower()
            
            # Setup image and coordinate information
            img_path = os.path.join(img_save_path, f"{sample['image_id']}.png")
            short_img_path = img_path[img_path.find(self.config.DATASET_NAME):]
            width, height = sample['image_size']['width'], sample['image_size']['height']
            
            # Process action based on type
            action_result = self._process_action_by_type(
                action_name, action_info, sample, width, height, step_instructions
            )
            
            if not action_result:
                return None
            
            action_str, action_refexp, task_attr, history = action_result
            
            # Generate action history
            history_str = self._generate_action_history(step_instructions, step_idx)
            
            # Create the final sample
            return self._create_action_sample(
                sample, action_str, action_refexp, task_attr, history_str,
                step_instructions, short_img_path, width, height, step_idx
            )
            
        except Exception as e:
            print(f"Error processing action step: {e}")
            return None
    
    def _extract_action_info(self, sample: Dict) -> Optional[Dict[str, Any]]:
        """Extract action information based on device type."""
        if self.config.CURRENT_DEVICE_TYPE == 'smartphone':
            return sample['actions_label']
        else:  # web
            actions = sample['actions_label']
            return actions[0] if actions else None
    
    def _process_action_by_type(self, action_name: str, action_info: Dict, sample: Dict,
                              width: int, height: int, step_instructions: List[str]) -> Optional[Tuple[str, str, Dict, str]]:
        """Process action based on its type and return formatted action data."""
        task_attr = {'original_action_type': action_name}
        
        if action_name == 'scroll':
            return self._process_scroll_action(action_info, task_attr, step_instructions)
        elif action_name == 'swipe':
            return self._process_swipe_action(action_info, task_attr, step_instructions)
        elif action_name == 'click':
            return self._process_click_action(action_info, sample, task_attr, step_instructions)
        elif action_name == 'tap':
            return self._process_tap_action(action_info, sample, task_attr, step_instructions)
        elif action_name == 'input':
            return self._process_input_action(action_info, sample, task_attr, step_instructions, width, height)
        elif action_name == 'hover':
            return self._process_hover_action(action_info, task_attr, step_instructions)
        elif action_name == 'enter':
            return self._process_enter_action(action_info, task_attr, step_instructions)
        elif action_name == 'answer':
            return self._process_answer_action(action_info, task_attr, step_instructions)
        elif action_name == 'select_text':
            return self._process_select_text_action(action_info, task_attr, step_instructions)
        elif action_name == 'select':
            return self._process_select_action(action_info, task_attr, step_instructions)
        elif action_name == 'copy':
            return self._process_copy_action(task_attr, step_instructions)
        else:
            print(f"Unknown action type: {action_name}")
            return None
    
    def _process_scroll_action(self, action_info: Dict, task_attr: Dict, step_instructions: List[str]) -> Optional[Tuple[str, str, Dict, str]]:
        """Process scroll action."""
        scroll_data = action_info['scroll']['related']
        down, right = float(scroll_data['down']), float(scroll_data['right'])
        down_abs, right_abs = abs(down), abs(right)
        
        # Skip horizontal scrolling in training
        if (self.config.CURRENT_SPLIT == 'train' and 
            not (down_abs > 0.01 and right_abs <= 0.05 and down_abs > right_abs)):
            return None
        
        direction = 'down' if down > 0 else 'up'
        distance = discretize_distance(abs(down))
        
        task_attr['direction'] = direction
        action_str = SCROLL_TEMPLATE.format(direction=direction, distance=distance)
        action_refexp = random.choice(SWIPE_PHRASES).format(direction=direction).replace('Swipe', 'Scroll')
        step_instructions.append(action_refexp)
        
        return action_str, action_refexp, task_attr, 'Step 1. Navigate to find the target content.'
    
    def _process_swipe_action(self, action_info: Dict, task_attr: Dict, step_instructions: List[str]) -> Optional[Tuple[str, str, Dict, str]]:
        """Process swipe action."""
        dual_points = action_info['dual_point']['related']
        from_point = list(map(float, dual_points['from'][7:-8].split(',')))
        to_point = list(map(float, dual_points['to'][7:-8].split(',')))
        
        vertical_shift = to_point[1] - from_point[1]
        horizontal_shift = to_point[0] - from_point[0]
        vertical_shift_abs = abs(vertical_shift)
        horizontal_shift_abs = abs(horizontal_shift)
        
        shift_ratio = vertical_shift_abs / (horizontal_shift_abs + 1e-6)
        if 2.4 > shift_ratio > 0.39:
            return None
        
        # Determine scrolling direction
        if abs(vertical_shift) > abs(horizontal_shift):
            direction = 'down' if vertical_shift > 0 else 'up'
            distance = discretize_distance(abs(vertical_shift))
        else:
            direction = 'right' if horizontal_shift > 0 else 'left'
            distance = discretize_distance(abs(horizontal_shift))
        
        start = list(map(lambda arg: round(arg * self.config.COORDINATE_SCALE), from_point))
        
        task_attr['direction'] = direction
        action_str = SWIPE_TEMPLATE.format(start_x=start[0], start_y=start[1], direction=direction, distance=distance)
        action_refexp = random.choice(SWIPE_PHRASES).format(direction=direction)
        step_instructions.append(action_refexp)
        
        history = 'Step 1. I have navigated to a page which I should explore to find the target content required by the task.'
        return action_str, action_refexp, task_attr, history
    
    def _process_click_action(self, action_info: Dict, sample: Dict, task_attr: Dict, step_instructions: List[str]) -> Optional[Tuple[str, str, Dict, str]]:
        """Process click action."""
        coords = list(map(float, action_info['element']['related'][5:-6].split(',')))
        x1, y1, x2, y2 = coords
        task_attr['bbox'] = coords
        
        center_x = round((x1 + x2) / 2 * self.config.COORDINATE_SCALE)
        center_y = round((y1 + y2) / 2 * self.config.COORDINATE_SCALE)
        
        if not (center_x < self.config.COORDINATE_SCALE and center_y < self.config.COORDINATE_SCALE):
            return None
        
        action_str = CLICK_TEMPLATE.format(target_x=center_x, target_y=center_y)
        action_refexp = random.choice(ACTION_PREFIXES['click']['specific']) + ' the task-related element'
        step_instructions.append(action_refexp)
        
        # Determine history based on previous action
        last_action = sample['actions_history'].split(' ')[-1].strip()
        if last_action == 'input':
            history = 'Step 1. Type texts into a text field.'
        else:
            history = 'Step 1. Go to a page to find the necessary element I should click on.'
        
        # Make extra intent grounding samples if enabled
        if sample.get('thoughts') and self.config.ENABLE_INTENT_GND:
            if random.random() <= self.config.BOUNDING_BOX_PROBABILITY:
                with_box = True
                x1, y1, x2, y2 = list(map(lambda arg: max(0, min(self.config.COORDINATE_SCALE-1, round(arg * self.config.COORDINATE_SCALE))), [x1, y1, x2, y2]))
                target = f'({x1},{y1},{x2},{y2})'
            else:
                center_x = max(0, min(self.config.COORDINATE_SCALE-1, round((x1+x2)/2 * self.config.COORDINATE_SCALE)))
                center_y = max(0, min(self.config.COORDINATE_SCALE-1, round((y1+y2)/2 * self.config.COORDINATE_SCALE)))
                target = f'({center_x},{center_y})'
                with_box = False
            
            # Note: Intent grounding sample creation would be handled separately
            self.statistics['intent_grounding_count'] += 1
        
        return action_str, action_refexp, task_attr, history
    
    def _process_tap_action(self, action_info: Dict, sample: Dict, task_attr: Dict, step_instructions: List[str]) -> Optional[Tuple[str, str, Dict, str]]:
        """Process tap action."""
        center = list(map(float, action_info['point']['related'][7:-8].split(',')))
        task_attr['center'] = center
        normalized_center = list(map(lambda x: max(0, min(self.config.COORDINATE_SCALE-1, round(x*self.config.COORDINATE_SCALE))), center))
        
        action_str = CLICK_TEMPLATE.format(target_x=normalized_center[0], target_y=normalized_center[1])
        action_refexp = random.choice(ACTION_PREFIXES['click']['specific']) + ' a task-related element'
        step_instructions.append(action_refexp)
        
        last_action = sample['actions_history'].split(' ')[-1].strip()
        if last_action == 'input':
            history = 'Step 1. Type texts into a text field.'
        else:
            history = 'Step 1. Go to a page to find the necessary element I should click on.'
        
        return action_str, action_refexp, task_attr, history
    
    def _process_input_action(self, action_info: Dict, sample: Dict, task_attr: Dict, step_instructions: List[str], width: int, height: int) -> Optional[Tuple[str, str, Dict, str]]:
        """Process input action."""
        text = action_info['text'].strip(' \n\\').replace('"', '\\"')
        task_attr['text'] = text
        
        is_smartphone = self.config.CURRENT_DEVICE_TYPE == 'smartphone'
        
        if is_smartphone:
            action_str = INPUT_TEMPLATE.format(text=text)
            textfield_desc = ''
        else:
            textfield_box = list(map(int, action_info['element']['absolute'][5:-6].split(', ')))
            textfield_center_x = max(0, min(self.config.COORDINATE_SCALE-1, round((textfield_box[0]+textfield_box[2])/2/width*self.config.COORDINATE_SCALE)))
            textfield_center_y = max(0, min(self.config.COORDINATE_SCALE-1, round((textfield_box[1]+textfield_box[3])/2/height*self.config.COORDINATE_SCALE)))
            action_str = INPUT_TARGET_TEMPLATE.format(target_x=textfield_center_x, target_y=textfield_center_y, text=text)
            
            elem_path = sample['actions_label'][0]['element_path']
            textfield_desc = elem_path[3:elem_path.find('")')+1] + ' '
        
        action_refexp = random.choice(INPUT_ACTION_PREFIXES_WITH_TEXT['specific']).format(text=text, target="the text box")
        step_instructions.append(action_refexp)
        history = f'Step 1: Focus on the {textfield_desc}text field to input texts.'
        
        return action_str, action_refexp, task_attr, history
    
    def _process_hover_action(self, action_info: Dict, task_attr: Dict, step_instructions: List[str]) -> Optional[Tuple[str, str, Dict, str]]:
        """Process hover action."""
        coords = list(map(float, action_info['element']['related'][5:-6].split(',')))
        x1, y1, x2, y2 = coords
        task_attr['bbox'] = coords
        relative_area = (x2-x1) * (y2-y1)
        
        # Skip certain hover actions in training
        if self.config.CURRENT_SPLIT == 'train' and (relative_area >= 0.65 or relative_area <= 0.001):
            return None
        
        normalized_center = [round((x1+x2)/2 * self.config.COORDINATE_SCALE), round((y1+y2)/2 * self.config.COORDINATE_SCALE)]
        
        if not (normalized_center[0] < self.config.COORDINATE_SCALE and normalized_center[1] < self.config.COORDINATE_SCALE):
            return None
        
        action_str = HOVER_TEMPLATE.format(target_x=normalized_center[0], target_y=normalized_center[1])
        action_refexp = random.choice(ACTION_PREFIXES['hover']['specific']) + ' an task-related element'
        step_instructions.append(action_refexp)
        
        history = 'Step 1. Go to a page to find the necessary element I should hover over.'
        return action_str, action_refexp, task_attr, history
    
    def _process_enter_action(self, action_info: Dict, task_attr: Dict, step_instructions: List[str]) -> Optional[Tuple[str, str, Dict, str]]:
        """Process enter key action."""
        action_str = PRESSKEY_TEMPLATE.format(key='Enter')
        action_refexp = random.choice(PRESSKEY_PREFIXES['Enter'])
        step_instructions.append(action_refexp)
        
        history = 'Step 1. Input a text query into the focused text field.'
        return action_str, action_refexp, task_attr, history
    
    def _process_answer_action(self, action_info: Dict, task_attr: Dict, step_instructions: List[str]) -> Optional[Tuple[str, str, Dict, str]]:
        """Process answer action."""
        task_attr['answer'] = ""
        
        if action_info["text"] == 'task complete':
            history = "Step 1. Navigate to the destination screen specified by the user's task"
            action_str = STATUS_TEMPLATE.format(goal_status="successful", answer="")
            task_attr['goal_status'] = 'successful'
            action_refexp = random.choice(TASK_STATUS_SENTENCES['successful'])
        elif action_info["text"] == 'task impossible':
            history = "Step 1. Navigate to the destination screen specified by the user's task"
            action_str = STATUS_TEMPLATE.format(goal_status="infeasible", answer="")
            task_attr['goal_status'] = 'infeasible'
            action_refexp = random.choice(TASK_STATUS_SENTENCES['infeasible'])
        else:
            history = "Step 1. Find the content that contains the answer."
            answer = action_info["text"].replace("\n", "\\n").replace('"', "'")
            action_str = STATUS_TEMPLATE.format(goal_status="successful", answer=answer)
            task_attr['goal_status'] = 'successful'
            action_refexp = random.choice(TASK_STATUS_SENTENCES['successful']) + " Answer based on the image information."
            task_attr['answer'] = action_info["text"]
        
        step_instructions.append(action_refexp)
        return action_str, action_refexp, task_attr, history
    
    def _process_select_text_action(self, action_info: Dict, task_attr: Dict, step_instructions: List[str]) -> Optional[Tuple[str, str, Dict, str]]:
        """Process select text action."""
        dual_points = action_info['dual_point']['related']
        from_point = list(map(float, dual_points['from'][7:-8].split(',')))
        to_point = list(map(float, dual_points['to'][7:-8].split(',')))
        task_attr['from'], task_attr['to'] = from_point, to_point
        
        norm_from_point = list(map(lambda arg: round(arg * self.config.COORDINATE_SCALE), from_point))
        norm_to_point = list(map(lambda arg: round(arg * self.config.COORDINATE_SCALE), to_point))
        
        if not (norm_from_point[0] < self.config.COORDINATE_SCALE and norm_from_point[1] < self.config.COORDINATE_SCALE and 
                norm_to_point[0] < self.config.COORDINATE_SCALE and norm_to_point[1] < self.config.COORDINATE_SCALE):
            return None
        
        history = "Step 1. Find the textual content I should select."
        action_refexp = random.choice(ACTION_PREFIXES['drag']['specific']) + ' the task-related texts'
        step_instructions.append(action_refexp)
        
        action_str = DRAG_TEMPLATE.format(start_x=norm_from_point[0], start_y=norm_from_point[1], 
                                         end_x=norm_to_point[0], end_y=norm_to_point[1])
        
        return action_str, action_refexp, task_attr, history
    
    def _process_select_action(self, action_info: Dict, task_attr: Dict, step_instructions: List[str]) -> Optional[Tuple[str, str, Dict, str]]:
        """Process select action."""
        coords = list(map(float, action_info['element']['related'][5:-6].split(',')))
        x1, y1, x2, y2 = coords
        normalized_center = [round((x1+x2)/2 * self.config.COORDINATE_SCALE), round((y1+y2)/2 * self.config.COORDINATE_SCALE)]
        
        history = "Step 1. Navigate to the screen that displays the element I should select."
        action_str = SELECT_TEMPLATE.format(target_x=normalized_center[0], target_y=normalized_center[1], value=action_info['text'])
        
        action_refexp = random.choice(SELECT_ACTION_PREFIXES_WITH_TEXT['specific']).format(
            text=action_info['text'].strip("'"), target="the menu"
        )
        step_instructions.append(action_refexp)
        
        return action_str, action_refexp, task_attr, history
    
    def _process_copy_action(self, task_attr: Dict, step_instructions: List[str]) -> Optional[Tuple[str, str, Dict, str]]:
        """Process copy action."""
        history = "Step 1. Find the textual content I should copy."
        task_attr['key_comb'] = 'ctrl-c'
        action_str = KEYCOMB_TEMPLATE.format(key_combination='ctrl+c')
        action_refexp = random.choice(KEYCOMB_PREFIXES['ctrl-c'])
        step_instructions.append(action_refexp)
        
        return action_str, action_refexp, task_attr, history
    
    def _generate_action_history(self, step_instructions: List[str], current_step: int) -> str:
        """Generate formatted action history for the current step."""
        if not step_instructions:
            return 'None'
        
        # Keep only unique previous actions up to the maximum limit
        _, clean_instructions = keep_unique_actions(step_instructions[:current_step])
        retained_history = clean_instructions[-self.config.MAX_PREVIOUS_ACTIONS:]
        
        if not retained_history:
            return 'None'
        
        # Format as numbered steps
        start_step = max(1, len(clean_instructions) - self.config.MAX_PREVIOUS_ACTIONS + 1)
        history_parts = []
        
        for i, instruction in enumerate(retained_history, start=start_step):
            formatted_instruction = instruction.strip(' .')
            history_parts.append(f"Step {i}. {formatted_instruction}.")
        
        return ' '.join(history_parts)
    
    def _create_action_sample(self, sample: Dict, action_str: str, action_refexp: str,
                            task_attr: Dict, history_str: str, step_instructions: List[str],
                            short_img_path: str, width: int, height: int, step_idx: int) -> Dict[str, Any]:
        """Create the final action sample in the required format."""
        # Parse action for metadata
        action = ast.literal_eval(action_str)
        action_type = action['action_type']
        
        # Add action reference expression if enabled
        if self.config.USE_ACTION_REFEXP:
            action_str = f"{QWEN_OBJ_REF_TAG_START}{action_refexp}{QWEN_OBJ_REF_TAG_END}\n{action_str}"
        
        # Determine device tag
        device_tag = 'Android' if self.config.CURRENT_DEVICE_TYPE == 'smartphone' else 'Web'
        
        # Create sample based on device type
        if self.config.CURRENT_DEVICE_TYPE == 'smartphone':
            processed_sample = make_actionplanning_sample(
                task_id=f"autogui_GUIAct-{sample['task_tag']}_planning_{sample['sample_id']}",
                global_task=sample['question'],
                history=history_str,
                gt_action='Action: ' + action_str,
                with_cot=False,
                use_action_refexp=self.config.USE_ACTION_REFEXP,
                device_tag=device_tag
            )
        else:  # web
            sample_id = sample['sample_id']
            if 'single' in sample['task_tag']:
                sample_id += f'-task{step_idx}'
            
            processed_sample = make_actionplanning_sample_web(
                task_id=f"autogui_GUIAct-{sample['task_tag']}_planning_{sample_id}",
                global_task=sample['question'],
                history=history_str,
                gt_action='Action: ' + action_str,
                with_cot=False,
                use_action_refexp=self.config.USE_ACTION_REFEXP,
                device_tag=device_tag
            )
        
        # Add metadata
        action['step_idx'] = 0 if 'single' in sample['task_tag'] else step_idx
        
        processed_sample.update({
            'task': sample['question'],
            'action_type': action_type,
            'task_attr': task_attr,
            'history': step_instructions[:-1],
            'step_instruction': step_instructions[-1],
            'action_refexp': action_refexp,
            'image': short_img_path,
            'step_info': action,
            'wxh': f"{width}x{height}",
            'device': 'web' if self.config.CURRENT_DEVICE_TYPE == 'web' else 'mobile'
        })
        
        return processed_sample
    
    def _generate_statistics_report(self) -> None:
        """Generate and print processing statistics."""
        # Calculate action type distribution
        action_stats = defaultdict(int)
        for sample in self.processed_samples:
            if 'planning' in sample['id']:
                match = re.search(r'"action_type":\s*"([^"]+)"', sample['conversations'][1]['value'])
                if match:
                    action_stats[match.group(1)] += 1
        
        print(f"""
GUIAct Processing Report:
------------------------
Total samples: {len(self.processed_samples)}
Action planning count: {self.statistics['action_planning_count']}
Intent grounding count: {self.statistics['intent_grounding_count']}
Action distribution: {dict(action_stats)}
        """)

# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def save_dataset_results(samples: List[Dict[str, Any]], dataset_name: str, 
                        config: DatasetConfig) -> None:
    """
    Save processed dataset samples and generate statistics.
    
    Args:
        samples: List of processed samples
        dataset_name: Name of the dataset (for filename)
        config: Configuration object
    """
    if not samples:
        print(f"No samples to save for {dataset_name}")
        return
    
    # Generate filename
    filename_parts = [dataset_name.lower(), config.CURRENT_SPLIT]
    
    if dataset_name == 'GUIAct':
        filename_parts.extend([config.CURRENT_DEVICE_TYPE])
        if config.USE_ACTION_REFEXP:
            filename_parts.append('wActRef')
    
    filename_parts.extend([f's{config.COORDINATE_SCALE}', str(len(samples))])
    filename = '-'.join(filename_parts) + '.json'
    
    save_path = os.path.join(config.SAVE_DIR, filename)
    
    # Save main dataset
    with open(save_path, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    # Save sample subset
    sample_size = min(len(samples), 256)
    sample_path = save_path.replace('.json', '_sample.json')
    with open(sample_path, "w") as f:
        json.dump(random.sample(samples, sample_size), f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(samples)} samples to {save_path}")

def main() -> None:
    """Main execution function for the GUICourse data processing pipeline."""
    print("Starting GUICourse Dataset Processing Pipeline...")
    print("=" * 60)
    
    # Initialize configuration and environment
    config = DatasetConfig()
    setup_environment()
    
    print(f"Configuration:")
    print(f"  Dataset: {config.DATASET_NAME}")
    print(f"  Split: {config.CURRENT_SPLIT}")
    print(f"  Device: {config.CURRENT_DEVICE_TYPE}")
    print(f"  Data Root: {config.DATA_ROOT}")
    print(f"  Save Directory: {config.SAVE_DIR}")
    print(f"  Coordinate Scale: {config.COORDINATE_SCALE}")
    print("=" * 60)
    
    # Process GUIEnv dataset
    if 'GUIEnv' in config.SUBTASK:  # Set to True to enable GUIEnv processing
        print("\nProcessing GUIEnv Dataset...")
        gui_env_processor = GUIEnvProcessor(config)
        gui_env_samples = gui_env_processor.process_dataset()
        save_dataset_results(gui_env_samples, 'GUIEnv', config)
    
    # Process GUIChat dataset
    if 'GUIChat' in config.SUBTASK:  # Set to True to enable GUIChat processing
        print("\nProcessing GUIChat Dataset...")
        gui_chat_processor = GUIChatProcessor(config)
        gui_chat_samples = gui_chat_processor.process_dataset()
        save_dataset_results(gui_chat_samples, 'GUIChat', config)
    
    # Process GUIAct dataset
    if 'GUIAct' in config.SUBTASK:  # Currently enabled
        print("\nProcessing GUIAct Dataset...")
        gui_act_processor = GUIActProcessor(config)
        gui_act_samples = gui_act_processor.process_dataset()
        save_dataset_results(gui_act_samples, 'GUIAct', config)
    
    print("\n" + "=" * 60)
    print("GUICourse Dataset Processing Pipeline Complete!")

if __name__ == "__main__":
    main()