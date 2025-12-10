"""
MultiUI Dataset Preprocessing Utilities.

This module processes the MultiUI dataset to generate training samples for multiple
UI understanding tasks. The dataset consists of UI screenshots paired with conversation
data that includes GPT-generated responses containing bounding box coordinates.

Supported Tasks:
1) UI Captioning: Generate overall UI descriptions and embedded captions
2) WebQA: Question-answering with bounding box localization
3) OCR: Extract text from UI elements (long text and titles)
4) Grounding: Locate UI elements and actions based on natural language descriptions

Key Processing Steps:
- Load raw MultiUI conversation data from JSON files
- Filter and validate samples based on task types and data integrity
- Extract and normalize bounding box coordinates from GPT responses
- Generate structured training samples with normalized coordinates
- Save processed data in chunks with statistics and sample subsets

The preprocessing pipeline ensures data quality by:
- Validating image file existence and dimensions
- Checking bounding box coordinate validity
- Filtering out samples with extraction failures
- Normalizing coordinates to a consistent scale
- Providing detailed statistics and error tracking

Configuration:
- Coordinate normalization scale (default: 1000)
- Task enable/disable flags for selective processing
- File paths for input/output directories
- Processing options like skipping integrity checks

Output:
- Processed training samples in JSON format
- Statistics file with processing metrics
- Sample subsets for inspection and debugging
"""

from __future__ import annotations

import json
import os
import random
import re
import traceback
from typing import Dict, List, Optional, Tuple

import magic
from tqdm import tqdm

from utils.data_utils.task_prompt_lib import (
    make_intentgnd_sample,
    web_loca_all_point_prompt,
    WITHBOX_TAG
)

# =============================================================================
# Configuration Constants
# =============================================================================

# Dataset identification
DATASET_NAME: str = "MultiUI"

# Task type categories for MultiUI dataset
# Maps high-level task categories to their specific task identifiers
MULTIUI_TASK_TYPES = {
    'UI understanding': ['meta_generate', 'embed_caption', 'action_prediction', 'webqa', 'embed_qa'],
    'OCR': ['long_text_OCR', 'title_identification'],
    'Grounding': ['element_ground', 'action_ground', 'element_ground_bbox', 'action_ground_bbox'],
    'Grounding_None': ['action_ground_{idx}_none_of_above', 'element_ground_{idx}_none_of_above',
                      'action_ground_bbox_{idx}_none_of_above', 'element_ground_bbox_{idx}_none_of_above']
}

# Coordinate normalization parameters
SCALE: int = 1000  # Normalization scale for bounding box coordinates

# Task processing flags - enable/disable specific task processing
ENABLE_UI_CAPTION: bool = False     # UI description generation tasks
ENABLE_WEBQA: bool = True          # Web-based question answering tasks
ENABLE_OCR: bool = True            # Optical character recognition tasks
ENABLE_GROUNDING: bool = True       # Element and action grounding tasks

# Processing control flags
SKIP_INTEGRITY_CHECKS: bool = False  # Skip sample validation and integrity checks

# =============================================================================
# File Paths Configuration
# =============================================================================

# Input data paths
MULTIUI_SAMPLE_FILE: str = "/mnt/vdb1/hongxin_li/MultiUI/stage1_data.json"
IMG_DIR: str = f"/mnt/vdb1/hongxin_li/{DATASET_NAME}"

# Output paths
SAVE_ROOT: str = f"/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/{DATASET_NAME}_processed"
STATS_FILE: str = os.path.join(SAVE_ROOT, "stats.json")

# =============================================================================
# Processing Parameters
# =============================================================================

# Task types to skip during processing (deprecated or problematic tasks)
SKIPPED_TASKS: List[str] = ['action_prediction', 'none_of_above']

# Sample size for inspection files
INSPECTION_SAMPLE_SIZE: int = 160

# =============================================================================
# Regular Expressions
# =============================================================================

# Pattern for extracting bounding box coordinates from GPT responses
# Matches format: [x1, y1, x2, y2] with optional whitespace
BBOX_PATTERN = re.compile(r'\[\s*(\d+\.\d+)\s*,\s*(\d+\.\d+)\s*,\s*(\d+\.\d+)\s*,\s*(\d+\.\d+)\s*\]')

# =============================================================================
# Directory Setup
# =============================================================================

# Ensure output directory exists
os.makedirs(SAVE_ROOT, exist_ok=True)


# =============================================================================
# Data Loading and Validation Functions
# =============================================================================

def load_multiui_dataset(sample_file_path: str) -> List[Dict]:
    """
    Load the raw MultiUI dataset from JSON file.

    Args:
        sample_file_path: Path to the JSON file containing MultiUI samples

    Returns:
        List of sample dictionaries from the dataset

    Raises:
        FileNotFoundError: If the sample file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(sample_file_path, 'r', encoding='utf-8') as f:
            multiui_data = json.load(f)
        print(f"Loaded {len(multiui_data)} samples from {sample_file_path}")
        return multiui_data
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"MultiUI sample file not found: {sample_file_path}") from exc
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in sample file: {e}", e.doc, e.pos)


def load_previous_invalid_samples() -> List[str]:
    """
    Load previously identified invalid sample IDs from stats file.

    Returns:
        List of invalid sample IDs to skip during processing
    """
    invalid_ids = []
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r', encoding='utf-8') as f:
                stats = json.load(f)
                invalid_ids = stats.get('invalid_elem', [])
                print(f"Loaded {len(invalid_ids)} previously invalid sample IDs")
        except (json.JSONDecodeError, KeyError):
            print("Warning: Could not load previous invalid samples, starting fresh")
            invalid_ids = []
    return invalid_ids


def generate_task_samples() -> None:
    """
    Generate sample examples for each task type from the MultiUI dataset.

    Categorizes samples by task type based on their IDs and saves the categorized
    examples to a JSON file for analysis and debugging purposes.

    This function is primarily used for dataset exploration and debugging.
    """
    # Load raw dataset
    multiui_data = load_multiui_dataset(MULTIUI_SAMPLE_FILE)

    # Initialize task categorization dictionary
    task_examples = {
        'meta_generate': [],       # Overall UI descriptions
        'embed_caption': [],       # Embedded UI element captions
        'action_prediction': [],   # Action prediction tasks
        'webqa': [],               # Web-based question answering
        'embed_qa': [],            # Embedded question answering
        'long_text_OCR': [],       # Long text OCR tasks
        'title_identification': [], # Title identification tasks
        'element_ground': [],      # Element grounding (no bbox)
        'action_ground': [],       # Action grounding (no bbox)
        'element_ground_bbox': [], # Element grounding with bbox
        'action_ground_bbox': [],  # Action grounding with bbox
    }

    remaining_samples = []
    processed_ids = set()

    # Categorize samples by task type
    for sample in tqdm(multiui_data, total=len(multiui_data), desc="Categorizing samples"):
        sample_id = sample['id']
        added = False

        for task_type in task_examples.keys():
            if task_type in sample_id:
                # Special handling for grounding tasks without bbox
                if 'ground' in task_type and 'bbox' not in task_type:
                    # Only include if 'bbox' is not in the sample ID
                    if 'bbox' not in sample_id:
                        processed_ids.add(sample_id)
                        task_examples[task_type].append(sample)
                        added = True
                else:
                    # Include all other matching task types
                    processed_ids.add(sample_id)
                    task_examples[task_type].append(sample)
                    added = True

        if not added:
            remaining_samples.append(sample)

    # Ensure all samples are categorized
    if remaining_samples:
        raise ValueError(f"Found {len(remaining_samples)} uncategorized samples. "
                        f"Sample IDs: {[s['id'] for s in remaining_samples[:5]]}")

    # Save categorized samples for inspection
    output_file = os.path.join(os.path.dirname(__file__), 'multiui_samples.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(task_examples, f, indent=2)

    print(f"Saved categorized samples to {output_file}")


# =============================================================================
# Coordinate Processing Functions
# =============================================================================

def convert_normalized_to_pixel_coords(
    normalized_box: List[float],
    image_height: int,
    image_width: int
) -> List[int]:
    """
    Convert normalized bounding box coordinates (0-1 range) to pixel coordinates.

    Args:
        normalized_box: Normalized coordinates [x1, y1, x2, y2] in range [0, 1]
        image_height: Image height in pixels
        image_width: Image width in pixels

    Returns:
        Pixel coordinates [x1, y1, x2, y2] as integers
    """
    x1, y1, x2, y2 = normalized_box
    pixel_x1 = round(x1 * image_width)
    pixel_y1 = round(y1 * image_height)
    pixel_x2 = round(x2 * image_width)
    pixel_y2 = round(y2 * image_height)
    return [pixel_x1, pixel_y1, pixel_x2, pixel_y2]


def normalize_bbox_coordinates_to_string(bbox_match: Tuple[str, ...]) -> str:
    """
    Convert bounding box coordinate strings to normalized format for training.

    Args:
        bbox_match: Regex match groups containing bbox coordinates as strings

    Returns:
        Normalized bounding box string in format '(x1,y1,x2,y2)' scaled by SCALE
    """
    # Convert string coordinates to float, then scale and round
    normalized_coords = [str(round(float(coord) * SCALE)) for coord in bbox_match]
    return f'({",".join(normalized_coords)})'


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """
    Extract image dimensions using the magic library.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (width, height) in pixels

    Raises:
        ValueError: If dimensions cannot be extracted from the file
    """
    try:
        # Use magic library to get image dimensions from file metadata
        dimension_info = re.search(r'(\d+) x (\d+)', magic.from_file(image_path))
        if dimension_info is None:
            raise ValueError(f"Could not extract dimensions from {image_path}")
        width, height = map(int, dimension_info.groups())
        return width, height
    except (OSError, ValueError) as e:
        raise ValueError(f"Failed to get dimensions for {image_path}: {e}") from e

def validate_sample_integrity() -> None:
    """
    Check the integrity of processed samples by visualizing bounding boxes.

    Loads sample data and draws bounding boxes on corresponding images to verify
    that the coordinate transformations are correct. This is primarily used for
    debugging and validation purposes.
    """
    try:
        import cv2  # Import here since it's only used for debugging
    except ImportError:
        print("OpenCV not available, cannot visualize bounding boxes")
        return

    sample_file = os.path.join(os.path.dirname(__file__), 'multiui_samples_10k.json')

    try:
        with open(sample_file, 'r', encoding='utf-8') as f:
            samples_by_task = json.load(f)
    except FileNotFoundError:
        print(f"Sample file not found: {sample_file}")
        return

    # Flatten all samples for processing
    all_samples = []
    for task_samples in samples_by_task.values():
        all_samples.extend(task_samples)

    print(f"Validating {len(all_samples)} samples...")

    for sample in all_samples[:5]:  # Limit to first 5 for debugging
        user_prompt = sample['conversations'][0]['value']
        gpt_response = sample['conversations'][1]['value']

        print(f'User: {user_prompt}\nGPT: {gpt_response}')

        img_path = os.path.join(IMG_DIR, sample['image'])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue

        height, width = img.shape[:2]

        # Extract and draw bounding boxes
        bbox_matches = BBOX_PATTERN.findall(gpt_response)
        for match in bbox_matches:
            # Convert normalized coords to pixel coords and draw
            pixel_bbox = convert_normalized_to_pixel_coords(
                [float(coord) for coord in match], height, width
            )
            cv2.rectangle(img, (pixel_bbox[0], pixel_bbox[1]),
                         (pixel_bbox[2], pixel_bbox[3]), (0, 255, 0), 2)

        cv2.imwrite('test.png', img)
        print("Visualization saved to test.png")
        break  # Only process first valid sample


def extract_query_from_user_prompt(user_prompt: str) -> Optional[str]:
    """
    Extract the target query text from a user prompt for grounding tasks.

    Parses the user prompt to find the quoted query text that describes the
    element or action to be located. Handles various quote types and formats,
    including smart quotes and different quotation styles.

    Args:
        user_prompt: The user's instruction text containing the query

    Returns:
        Extracted and cleaned query string, or None if extraction fails
    """
    if not user_prompt or not isinstance(user_prompt, str):
        return None

    # Normalize smart quotes to standard quotes
    normalized_prompt = (user_prompt
                        .replace('\u201D', '"')  # Right double quotation mark
                        .replace('\u201C', '"')  # Left double quotation mark
                        .replace('\u2018', "'")  # Left single quotation mark
                        .replace('\u2019', "'")
                        .replace("'s ", " ").replace("s' ", " ")) # remove possessive apostrophes

    # Determine quote type used in the prompt
    single_quote_pos = normalized_prompt.find("'")
    double_quote_pos = normalized_prompt.find('"')

    # Select the first occurring quote type
    if single_quote_pos != -1 and (double_quote_pos == -1 or single_quote_pos < double_quote_pos):
        quote_char = "'"
        quote_start_pos = single_quote_pos
    elif double_quote_pos != -1:
        quote_char = '"'
        quote_start_pos = double_quote_pos
    else:
        quote_char = None
        quote_start_pos = -1

    # Find the start position of the query text
    colon_pos = normalized_prompt.find(':')
    desc_quote_pos = normalized_prompt.find('description "')

    comma_pos = normalized_prompt.find(',')
    desc_pos = normalized_prompt[:comma_pos].find('description ')

    if desc_pos != -1 and not desc_pos < quote_start_pos < comma_pos: # Example: <image>\nGiven the element description Sanctuary Supply Depot, predict the ...
        query_start_pos = desc_pos + 11
    elif colon_pos == -1:
        query_start_pos = desc_quote_pos + 10 if desc_quote_pos != -1 else -1
    elif quote_start_pos != -1 and colon_pos > quote_start_pos:
        query_start_pos = colon_pos
    else:
        query_start_pos = colon_pos

    if query_start_pos == -1:
        return None

    # Extract the quoted query
    if quote_char:
        # Find matching quotes for the query
        left_quote_pos = normalized_prompt.find(quote_char, query_start_pos, query_start_pos + 4)
        if left_quote_pos == -1:
            return None

        right_quote_pos = normalized_prompt.rfind(quote_char)
        if right_quote_pos == left_quote_pos:
            # Single quoted segment
            query = normalized_prompt[:right_quote_pos + 1]
        else:
            # Extract text between quotes
            query = normalized_prompt[left_quote_pos + 1:right_quote_pos]
    else:
        # Fallback: extract until period if no quotes found
        period_pos = normalized_prompt.find('.', query_start_pos)

        if desc_pos != -1:
            query = normalized_prompt[desc_pos + 11:comma_pos]
        elif period_pos != -1:
            query = normalized_prompt[query_start_pos + 1:period_pos]
        else:
            query = normalized_prompt[query_start_pos + 1:]

    # Clean and validate the extracted query
    if not query or query in ['', '"', "'"]:
        return None

    # Remove surrounding quotes if present
    query = query.strip(' \'".')

    # Skip obviously invalid queries
    if any(invalid_pattern in query for invalid_pattern in ['="', "='", 'float']):
        return None

    return query


def process_ui_caption_sample(sample: Dict[str, str], task_counters: Dict[str, int]) -> Optional[Dict[str, str]]:
    """
    Process a UI caption sample for overall description or embedded caption tasks.

    Creates training samples for UI understanding tasks that involve generating
    textual descriptions of user interfaces. Handles both meta-level descriptions
    (entire UI) and embedded element captions.

    Args:
        sample: Raw sample dictionary containing conversations and metadata.
            Expected to have 'id' field for task type identification.
        task_counters: Dictionary tracking counts of processed samples by type.
            Keys: 'uicaption', 'embedcaption' for incrementing task IDs.

    Returns:
        Processed sample dictionary with updated task ID, or None if:
        - UI caption tasks are disabled (ENABLE_UI_CAPTION = False)
        - Sample doesn't match expected caption task types
        - Processing fails for any reason

    Task Types Handled:
        - 'meta_generate': Overall UI description generation
        - 'caption': Embedded UI element captioning
    """
    # Early return if UI caption tasks are disabled
    if not ENABLE_UI_CAPTION:
        return None

    sample_id = sample['id']

    # Determine task type and assign appropriate task ID
    if 'meta_generate' in sample_id:
        # Overall UI description generation task
        task_id = f'autogui_multiui_UIcaption_{task_counters["uicaption"]}'
        task_counters['uicaption'] += 1
    elif 'caption' in sample_id:
        # Embedded UI element captioning task
        task_id = f'autogui_multiui_embedcaption_{task_counters["embedcaption"]}'
        task_counters['embedcaption'] += 1
    else:
        # Sample doesn't match expected caption task types
        return None

    # Create processed sample with updated task identifier
    processed_sample = sample.copy()
    processed_sample['id'] = task_id

    return processed_sample


def process_webqa_sample(sample: Dict[str, str], task_counters: Dict[str, int]) -> Optional[Dict[str, str]]:
    """
    Process a WebQA sample with question-answering and bounding box localization.

    Handles both web-based question answering tasks and embedded QA tasks,
    normalizing bounding box coordinates in GPT responses for training. Extracts
    and scales bounding box coordinates from GPT-generated responses.

    Args:
        sample: Raw sample dictionary with conversations and metadata.
            Expected to have 'id' and 'conversations' fields.
        task_counters: Dictionary tracking counts of processed samples by type.
            Keys: 'webqa', 'embed_qa' for incrementing task IDs.

    Returns:
        Processed sample dictionary with normalized coordinates, or None if:
        - WebQA tasks are disabled (ENABLE_WEBQA = False)
        - No bounding boxes found in GPT response (for webqa tasks)
        - Sample doesn't match expected WebQA task types

    Task Types Handled:
        - 'webqa': Web-based question answering with bbox localization
        - 'embed_qa': Embedded question answering (no bbox processing)

    Processing Details:
        - Extracts bbox coordinates using BBOX_PATTERN regex
        - Normalizes coordinates using SCALE factor
        - Replaces original bbox strings with normalized versions
    """
    # Early return if WebQA tasks are disabled
    if not ENABLE_WEBQA:
        return None

    sample_id = sample['id']
    gpt_response = sample['conversations'][1]['value']

    if 'webqa' in sample_id:
        # Process WebQA samples with bounding box localization
        bbox_matches = BBOX_PATTERN.findall(gpt_response)
        if bbox_matches:
            # Replace normalized bbox coordinates with scaled coordinates for training
            for match in bbox_matches:
                original_bbox_str = f'[{", ".join(match)}]'
                normalized_bbox_str = normalize_bbox_coordinates_to_string(match)
                gpt_response = gpt_response.replace(original_bbox_str, normalized_bbox_str)

            # Create processed sample with updated response and unique task ID
            processed_sample = sample.copy()
            processed_sample['conversations'][1]['value'] = gpt_response
            processed_sample['id'] = f'autogui_multiui_webqa_{task_counters["webqa"]}'
            task_counters['webqa'] += 1
            return processed_sample
        else:
            # No bounding boxes found in WebQA response - cannot process
            return None

    elif 'embed_qa' in sample_id:
        # Process embedded QA samples (no bbox processing needed)
        processed_sample = sample.copy()
        processed_sample['id'] = f'autogui_multiui_embedqa_{task_counters["embed_qa"]}'
        task_counters['embed_qa'] += 1
        return processed_sample
    else:
        1+1
    # Sample doesn't match expected WebQA task types
    return None


def process_ocr_sample(sample: Dict[str, str], task_counters: Dict[str, int]) -> Optional[Dict[str, str]]:
    """
    Process an OCR-related sample for text extraction tasks.

    Handles optical character recognition tasks for extracting text content
    from UI elements, including both long text blocks and title elements.
    No coordinate processing is needed for OCR tasks.

    Args:
        sample: Raw sample dictionary with conversations and metadata.
            Expected to have 'id' field for task type identification.
        task_counters: Dictionary tracking counts of processed samples by type.
            Keys: 'ocr', 'titleocr' for incrementing task IDs.

    Returns:
        Processed sample dictionary with updated task ID, or None if:
        - OCR tasks are disabled (ENABLE_OCR = False)
        - Sample doesn't match expected OCR task types

    Task Types Handled:
        - 'OCR': Long text OCR tasks (long_text_OCR)
        - 'title_identification': Title identification tasks

    Note:
        OCR tasks focus on text extraction rather than localization,
        so no bounding box processing is performed.
    """
    # Early return if OCR tasks are disabled
    if not ENABLE_OCR:
        return None

    sample_id = sample['id']

    if 'OCR' in sample_id:
        # Process long text OCR samples - text extraction from UI elements
        processed_sample = sample.copy()
        processed_sample['id'] = f'autogui_multiui_ocr_{task_counters["ocr"]}'
        task_counters['ocr'] += 1
        return processed_sample

    elif 'title_identification' in sample_id:
        # Process title identification samples - identifying UI titles
        processed_sample = sample.copy()
        processed_sample['id'] = f'autogui_multiui_titleidentification_{task_counters["titleocr"]}'
        task_counters['titleocr'] += 1
        return processed_sample

    # Sample doesn't match expected OCR task types
    return None


def process_grounding_sample(
    sample: Dict[str, str],
    invalid_sample_ids: List[str],
    task_counters: Dict[str, int]
) -> Optional[Dict[str, str]]:
    """
    Process a grounding sample with bounding box localization and intent extraction.

    Handles both element grounding (locating UI elements by description) and action
    grounding (locating elements for performing actions). Performs coordinate
    normalization, query extraction from user prompts, and validity checking.

    Args:
        sample: Raw sample dictionary with conversations and metadata.
            Expected to have 'id', 'image', and 'conversations' fields.
        invalid_sample_ids: List of previously identified invalid sample IDs to skip.
            Used to avoid re-processing samples known to be problematic.
        task_counters: Dictionary tracking counts of processed samples by type.
            Keys: 'elemgnd', 'intentgnd', 'invalid_elem' for tracking.

    Returns:
        Processed grounding sample dictionary with normalized coordinates and extracted intent,
        or None if processing fails or grounding tasks are disabled.

    Task Types Handled:
        - 'element_ground_bbox': Element grounding with bounding box localization
        - 'action_ground_bbox': Action grounding with bounding box localization

    Processing Steps:
        1. Validate sample ID and task enablement
        2. Extract and validate bounding box coordinates from GPT response
        3. Convert normalized coordinates to pixel coordinates
        4. Extract target query from user prompt
        5. Generate appropriate sample format based on grounding type
        6. Update user prompt with grounding instruction

    Validation Checks:
        - Image file existence and dimension extraction
        - Bounding box coordinate validity
        - Query extraction success
        - Duplicate invalid sample filtering
    """
    # Skip processing if grounding tasks are disabled
    if not ENABLE_GROUNDING:
        return None

    sample_id = sample['id']

    # Only process samples with bounding box grounding
    if not ('element_ground_bbox' in sample_id or 'action_ground_bbox' in sample_id):
        return None

    # Skip previously identified invalid samples
    if sample_id in invalid_sample_ids:
        task_counters['invalid_elem'].append(sample_id)
        return None

    gpt_response = sample['conversations'][1]['value']
    user_prompt = sample['conversations'][0]['value']

    # Extract bounding box coordinates from GPT response
    bbox_matches = BBOX_PATTERN.findall(gpt_response)
    if not bbox_matches:
        return None

    bbox_match = bbox_matches[0]  # Use first bounding box found

    # Get image dimensions for coordinate conversion
    img_path = os.path.join(os.path.dirname(IMG_DIR), sample['image'])
    try:
        image_width, image_height = get_image_dimensions(img_path)
    except ValueError:
        return None

    # Convert normalized coordinates to pixel coordinates
    pixel_bbox = convert_normalized_to_pixel_coords(
        [float(coord) for coord in bbox_match], image_height, image_width
    )

    # TODO: Add pure color check for invalid elements if needed
    # img = cv2.imread(img_path)
    # if img is not None and is_pure_color(img, pixel_bbox):
    #     task_counters['invalid_elem'].append(sample_id)
    #     return None

    # Normalize coordinates for training
    normalized_bbox_str = normalize_bbox_coordinates_to_string(bbox_match)

    # Extract target query from user prompt
    query = extract_query_from_user_prompt(user_prompt)
    if query is None:
        task_counters['invalid_elem'].append(sample_id)
        return None

    # Process based on grounding task type
    if 'element_ground_bbox' in sample_id:
        # Element grounding: locate UI elements by description
        processed_sample = sample.copy()
        processed_sample['conversations'][1]['value'] = normalized_bbox_str

        # Update user prompt with grounding instruction
        processed_sample['conversations'][0]['value'] = re.sub(
            r'\s+', ' ',
            random.choice(web_loca_all_point_prompt).replace("with point", "with bbox") + f' {query.strip()}'
        )
        processed_sample['id'] = f'autogui_multiui_elemgnd_{task_counters["elemgnd"]}'
        processed_sample['task_attr'] = query
        processed_sample['unnormalized_box'] = pixel_bbox
        task_counters['elemgnd'] += 1
        return processed_sample

    else:  # action_ground_bbox
        # Action grounding: locate elements for performing actions
        intent_description = query[0].lower() + query[1:]  # Normalize intent casing

        grounding_sample = make_intentgnd_sample(
            task_id=f'autogui_multiui_intentgnd_{task_counters["intentgnd"]}',
            intent=intent_description.strip(' "'),
            loc=normalized_bbox_str,
            output_tag=WITHBOX_TAG
        )
        grounding_sample['image'] = sample['image']
        grounding_sample['task_attr'] = intent_description
        grounding_sample['unnormalized_box'] = pixel_bbox

        task_counters['intentgnd'] += 1
        return grounding_sample

def save_processed_data(
    caption_samples: List[Dict[str, str]],
    grounding_qa_samples: List[Dict[str, str]],
    processing_stats: Dict
) -> None:
    """
    Save processed samples and statistics to disk in organized files.

    Creates separate output files for different task categories (caption vs grounding/QA/OCR)
    and saves comprehensive processing statistics. Also generates smaller sample files
    for quick inspection and debugging purposes.

    Args:
        caption_samples: List of processed UI caption samples.
            Contains samples from meta_generate and embed_caption tasks.
        grounding_qa_samples: List of processed grounding, QA, and OCR samples.
            Contains samples from webqa, embed_qa, OCR, title_identification,
            element_ground_bbox, and action_ground_bbox tasks.
        processing_stats: Dictionary containing comprehensive processing statistics and metrics.
            Includes counts, error tracking, and performance metrics.

    Output Files Generated:
        - Grounding/QA/OCR samples: {DATASET_NAME}_gnd_ref_qa_scale{SCALE}_{count}k.json
        - UI Caption samples: {DATASET_NAME}_caption_{count}k.json
        - Sample subsets: *_sample.json files with INSPECTION_SAMPLE_SIZE samples
        - Statistics: stats.json with processing metrics

    File Organization:
        - Separate files for different task categories to enable selective loading
        - Consistent naming convention with dataset name, task type, and scale
        - Sample files for quick validation and debugging
    """
    # Save comprehensive processing statistics
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(processing_stats, f, indent=2)

    # Save grounding/QA samples (OCR, WebQA, Grounding tasks)
    if grounding_qa_samples:
        gnd_qa_filename = os.path.join(
            SAVE_ROOT,
            f'{DATASET_NAME}_gnd_ref_qa_scale{SCALE}_{len(grounding_qa_samples)//1000}k.json'
        )

        # Save representative sample subset for inspection
        sample_filename = gnd_qa_filename.replace('.json', '_sample.json')
        sample_size = min(len(grounding_qa_samples), INSPECTION_SAMPLE_SIZE)
        with open(sample_filename, 'w', encoding='utf-8') as f:
            json.dump(random.sample(grounding_qa_samples, sample_size), f, indent=2)

        # Save complete dataset
        with open(gnd_qa_filename, 'w', encoding='utf-8') as f:
            print(f"Saving {len(grounding_qa_samples)} grounding/QA samples to {gnd_qa_filename}")
            json.dump(grounding_qa_samples, f)

    # Save UI caption samples (UI description generation tasks)
    if caption_samples:
        caption_filename = os.path.join(
            SAVE_ROOT,
            f'{DATASET_NAME}_caption_{len(caption_samples)//1000}k.json'
        )

        # Save representative sample subset for inspection
        sample_filename = caption_filename.replace('.json', '_sample.json')
        sample_size = min(len(caption_samples), INSPECTION_SAMPLE_SIZE)
        with open(sample_filename, 'w', encoding='utf-8') as f:
            json.dump(random.sample(caption_samples, sample_size), f, indent=2)

        # Save complete dataset
        with open(caption_filename, 'w', encoding='utf-8') as f:
            print(f"Saving {len(caption_samples)} UI caption samples to {caption_filename}")
            json.dump(caption_samples, f)


def process_all_samples(
    multiui_data: List[Dict],
    invalid_ids: List[str],
    counters: Dict,
    caption_samples: List[Dict],
    gnd_ref_qa_samples: List[Dict]
) -> List[List]:
    """
    Process all samples in the MultiUI dataset according to their task types.

    Iterates through each sample, applies appropriate processing based on task type,
    handles exceptions, and collects results into appropriate output collections.

    Args:
        multiui_data: List of raw sample dictionaries to process
        invalid_ids: List of previously identified invalid sample IDs to skip
        counters: Dictionary of counters for tracking processing statistics
        caption_samples: List to collect processed UI caption samples
        gnd_ref_qa_samples: List to collect processed grounding/QA/OCR samples

    Returns:
        List of exception records, where each record contains [sample_idx, traceback]

    Processing Flow:
        1. Skip unwanted task types based on SKIPPED_TASKS
        2. Update image path to include dataset prefix
        3. Route sample to appropriate processing function based on task type
        4. Handle exceptions and track problematic samples
    """
    exception_sample_idxs = []

    # Process each sample with progress tracking
    for sample_idx, sample in tqdm(enumerate(multiui_data), total=len(multiui_data),
                                   desc="Processing MultiUI samples"):
        sample_id = sample['id']

        # Skip deprecated or problematic task types
        if any(skipped_task in sample_id for skipped_task in SKIPPED_TASKS):
            continue

        counters['processed'] += 1

        # Update image path to include dataset directory prefix
        sample['image'] = f"{DATASET_NAME}/{sample['image']}"

        try:
            # Route sample to appropriate processing function based on task type
            if 'meta_generate' in sample_id or 'caption' in sample_id:
                processed_sample = process_ui_caption_sample(sample, counters)
                if processed_sample:
                    caption_samples.append(processed_sample)

            elif 'webqa' in sample_id or 'embed_qa' in sample_id:
                processed_sample = process_webqa_sample(sample, counters)
                if processed_sample:
                    gnd_ref_qa_samples.append(processed_sample)

            elif 'OCR' in sample_id or 'title_identification' in sample_id:
                processed_sample = process_ocr_sample(sample, counters)
                if processed_sample:
                    gnd_ref_qa_samples.append(processed_sample)

            elif 'element_ground_bbox' in sample_id or 'action_ground_bbox' in sample_id:
                processed_sample = process_grounding_sample(sample, invalid_ids, counters)
                if processed_sample:
                    gnd_ref_qa_samples.append(processed_sample)

        except (ValueError, KeyError, IndexError) as e:
            # Track exceptions with sample index and full traceback for debugging
            exception_sample_idxs.append([sample_idx, traceback.format_exc()])
            print(f"Error processing sample {sample_idx} ({sample_id}): {e}")

    return exception_sample_idxs


def prepare_processing_statistics(
    multiui_data: List[Dict],
    counters: Dict,
    caption_samples: List[Dict],
    gnd_ref_qa_samples: List[Dict],
    exception_sample_idxs: List[List]
) -> Dict:
    """
    Prepare comprehensive processing statistics for reporting and debugging.

    Compiles all processing metrics, counts, and error tracking information
    into a structured dictionary for output and analysis.

    Args:
        multiui_data: Original raw dataset for total count
        counters: Processing counters with task-specific counts
        caption_samples: Processed UI caption samples
        gnd_ref_qa_samples: Processed grounding/QA/OCR samples
        exception_sample_idxs: List of exception records with sample indices and tracebacks

    Returns:
        Dictionary containing comprehensive processing statistics including:
        - Total and processed sample counts
        - Task-specific sample counts
        - Error tracking and invalid sample information
        - Exception details for debugging
    """
    return {
        'total_samples': len(multiui_data),
        'processed_samples': counters['processed'],
        'invalid_sample_cnt': len(counters['invalid_elem']),
        'collected_samples': len(caption_samples) + len(gnd_ref_qa_samples),
        'uicaption_cnt': counters['uicaption'],
        'embedcaption_cnt': counters['embedcaption'],
        'webqa_cnt': counters['webqa'],
        'embed_qa_cnt': counters['embed_qa'],
        'ocr_cnt': counters['ocr'],
        'titleocr_cnt': counters['titleocr'],
        'elemgnd_cnt': counters['elemgnd'],
        'intentgnd_cnt': counters['intentgnd'],
        'invalid_elem': counters['invalid_elem'],
        'exception_sample_idxs': exception_sample_idxs
    }


def print_processing_summary(stats: Dict) -> None:
    """
    Print comprehensive processing summary to console with detailed statistics.

    Displays key statistics about the dataset processing including sample counts,
    task distribution, data quality metrics, and processing ratios similar to WAE.

    Args:
        stats: Dictionary containing processing statistics from prepare_processing_statistics
    """
    print("\n" + "="*60)
    print("MULTIUI DATASET PROCESSING COMPLETE")
    print("="*60)

    # Calculate processing ratios
    processed_ratio = stats['processed_samples'] / stats['total_samples'] if stats['total_samples'] > 0 else 0.0
    valid_ratio = stats['collected_samples'] / stats['processed_samples'] if stats['processed_samples'] > 0 else 0.0
    invalid_ratio = stats['invalid_sample_cnt'] / stats['processed_samples'] if stats['processed_samples'] > 0 else 0.0

    print("Dataset Overview:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Processed samples: {stats['processed_samples']} ({processed_ratio:.2f})")
    print(f"  Valid collected samples: {stats['collected_samples']} ({valid_ratio:.2f})")
    print(f"  Invalid/problematic samples: {stats['invalid_sample_cnt']} ({invalid_ratio:.2f})")

    print("\nTask Distribution:")
    print(f"  UI Caption: {stats['uicaption_cnt']} | Embedded Caption: {stats['embedcaption_cnt']}")
    print(f"  WebQA: {stats['webqa_cnt']} | Embedded QA: {stats['embed_qa_cnt']}")
    print(f"  OCR: {stats['ocr_cnt']} | Title OCR: {stats['titleocr_cnt']}")
    print(f"  Element Grounding: {stats['elemgnd_cnt']} | Intent Grounding: {stats['intentgnd_cnt']}")

    if stats['exception_sample_idxs']:
        print("\nError Tracking:")
        print(f"  Samples with exceptions: {len(stats['exception_sample_idxs'])}")
        print("  Note: Check stats.json for detailed exception information")

    print("="*60 + "\n")


def initialize_processing_counters() -> Dict:
    """
    Initialize counters and data structures for tracking processing statistics.

    Returns:
        Dictionary containing all counters and tracking lists initialized to zero/empty.
        Keys include task counters, processing counters, and error tracking lists.
    """
    return {
        'uicaption': 0,      # UI caption task counter
        'embedcaption': 0,   # Embedded caption task counter
        'webqa': 0,          # WebQA task counter
        'embed_qa': 0,       # Embedded QA task counter
        'ocr': 0,            # OCR task counter
        'titleocr': 0,       # Title OCR task counter
        'elemgnd': 0,        # Element grounding task counter
        'intentgnd': 0,      # Intent grounding task counter
        'processed': 0,      # Total processed samples counter
        'invalid_elem': []   # List of invalid sample IDs
    }


def load_and_prepare_data() -> Tuple[List[Dict], List[str]]:
    """
    Load raw MultiUI dataset and previously identified invalid samples.

    Returns:
        Tuple containing:
        - multiui_data: List of raw sample dictionaries from the dataset
        - invalid_ids: List of previously identified invalid sample IDs to skip

    Raises:
        FileNotFoundError: If the sample file doesn't exist
        json.JSONDecodeError: If the sample file contains invalid JSON
    """
    # Load raw MultiUI dataset
    with open(MULTIUI_SAMPLE_FILE, 'r', encoding='utf-8') as f:
        multiui_data = json.load(f)

    # Load previously identified invalid samples for skipping
    invalid_ids = load_previous_invalid_samples()

    return multiui_data, invalid_ids


def make_multiui_data() -> None:
    """
    Main function to process MultiUI dataset and generate training samples.

    Orchestrates the complete data processing pipeline:
    1. Load raw dataset and previous processing state
    2. Initialize processing counters and data structures
    3. Process each sample according to its task type
    4. Apply filtering, validation, and coordinate normalization
    5. Save processed samples and comprehensive statistics

    The pipeline ensures data quality through validation, error tracking,
    and structured output organization for different task categories.
    """
    # Load raw data and previous processing state
    multiui_data, invalid_ids = load_and_prepare_data()

    # Initialize counters and collections for processing tracking
    counters = initialize_processing_counters()
    caption_samples = []      # UI caption task samples
    gnd_ref_qa_samples = []   # Grounding, QA, and OCR task samples
    exception_sample_idxs = []  # Track samples that caused exceptions

    # Process all samples through the main processing pipeline
    exception_sample_idxs = process_all_samples(
        multiui_data, invalid_ids, counters,
        caption_samples, gnd_ref_qa_samples
    )

    # Prepare comprehensive processing statistics
    stats = prepare_processing_statistics(
        multiui_data, counters, caption_samples, gnd_ref_qa_samples, exception_sample_idxs
    )

    # Save processed samples and statistics to disk
    save_processed_data(caption_samples, gnd_ref_qa_samples, stats)

    # Print comprehensive processing summary
    print_processing_summary(stats)



if __name__ == '__main__':
    make_multiui_data()