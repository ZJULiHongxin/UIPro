"""
UIPro Image Processing Module

This module provides comprehensive image processing utilities for the UIPro system,
including image slicing, patching, resizing, and various preprocessing operations
optimized for multi-modal GUI understanding tasks.
"""

import math
from typing import List, Tuple, Union
from PIL import Image
import torch
from torch.nn import functional as F
import torchvision.transforms.functional as TF


# =============================================================================
# Image Processing Constants
# =============================================================================

# Patch configuration for image tokenization
PATCH_SIZE: int = 14                                    # Size of each image patch
PATCH_NUM_WIDTH: int = 24                              # Number of patches horizontally
PATCH_NUM_HEIGHT: int = 24                             # Number of patches vertically
POSITION_EMBEDDING_LENGTH: int = 1024                  # Length of position embeddings

# Derived constants
MAX_PATCHES: int = PATCH_NUM_WIDTH * PATCH_NUM_HEIGHT  # Maximum number of patches (576)
TOKEN_LENGTH: int = 3 * PATCH_SIZE * PATCH_SIZE        # Length of each patch token
IMAGE_WIDTH: int = PATCH_SIZE * PATCH_NUM_WIDTH        # Standard image width (336)
IMAGE_HEIGHT: int = PATCH_SIZE * PATCH_NUM_HEIGHT      # Standard image height (336)

# =============================================================================
# Core Image Processing Functions
# =============================================================================

def torch_extract_patches(image_tensor: torch.Tensor, patch_height: int, patch_width: int) -> torch.Tensor:
    """
    Extract patches from a given image tensor using unfold operation.
    
    This utility function extracts non-overlapping patches from an image tensor,
    which is useful for vision transformer and patch-based processing.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor of shape (C, H, W)
        patch_height (int): Height of each patch to extract
        patch_width (int): Width of each patch to extract
        
    Returns:
        torch.Tensor: Extracted patches of shape (1, num_patches_h, num_patches_w, C*patch_h*patch_w)
    """

    image_tensor = image_tensor.unsqueeze(0)
    patches = torch.nn.functional.unfold(image_tensor, (patch_height, patch_width), stride=(patch_height, patch_width))
    patches = patches.reshape(image_tensor.size(0), image_tensor.size(1), patch_height, patch_width, -1)
    patches = patches.permute(0, 4, 2, 3, 1).reshape(
        image_tensor.size(2) // patch_height,
        image_tensor.size(3) // patch_width,
        image_tensor.size(1) * patch_height * patch_width,
    )
    return patches.unsqueeze(0)

def adapt_size(origin_height: int, origin_width: int, 
               patch_height: int = PATCH_SIZE, patch_width: int = PATCH_SIZE,
               max_patches: int = MAX_PATCHES) -> Tuple[int, int, int, int]:
    """
    Calculate optimal image dimensions for adaptive resizing while maintaining aspect ratio.
    
    This function computes the best resize dimensions that fit within the patch limit
    while preserving the original aspect ratio as much as possible.
    
    Args:
        origin_height (int): Original image height
        origin_width (int): Original image width  
        patch_height (int): Height of each patch (default: PATCH_SIZE)
        patch_width (int): Width of each patch (default: PATCH_SIZE)
        max_patches (int): Maximum number of patches allowed (default: MAX_PATCHES)
        
    Returns:
        Tuple[int, int, int, int]: 
            - resized_height: Interpolated image height
            - resized_width: Interpolated image width
            - resized_patch_height_num: Number of vertical patches
            - resized_patch_width_num: Number of horizontal patches
    """
    scale = math.sqrt(max_patches * (patch_height / origin_height) * (patch_width / origin_width))
    resized_patch_height_num = max(min(math.floor(scale * origin_height / patch_height), max_patches), 1)
    resized_patch_width_num = max(min(math.floor(scale * origin_width / patch_width), max_patches), 1)
    resized_height = max(resized_patch_height_num * PATCH_SIZE, 1)
    resized_width = max(resized_patch_width_num * PATCH_SIZE, 1)
    return resized_height, resized_width, resized_patch_height_num, resized_patch_width_num

def cal_num_of_slices(origin_image_width: int, origin_image_height: int) -> Tuple[int, int]:
    """
    Calculate the optimal number of slices for image partitioning.
    
    This function determines how to slice an image into smaller parts while
    maintaining aspect ratio similarity and staying within processing limits.
    
    Args:
        origin_image_width (int): Original image width in pixels
        origin_image_height (int): Original image height in pixels
        
    Returns:
        Tuple[int, int]: Optimal number of slices (width_slices, height_slices)
    """
    # Calculate scale factor based on image area vs standard area
    scale = origin_image_width * origin_image_height / (IMAGE_WIDTH * IMAGE_HEIGHT)  
    scale = math.ceil(scale)
    
    # Limit maximum scale to 6 for computational efficiency
    if scale > 6:
        scale = 6
    
    def factorize(n: int) -> List[Tuple[float, int, int]]:
        """Generate factorization ratios for a given number."""
        factors = []
        for i in range(1, n + 1):
            if n % i == 0:
                factors.append((i / (n / i), i, n // i))
        return factors
    
    # Pre-compute factorizations for possible scales
    numbers = [1, 2, 3, 4, 5, 6, 7]
    factor_dict = {num: factorize(num) for num in numbers}
    
    # Calculate original aspect ratio in log space for better comparison
    log_origin_ratio = math.log(origin_image_width / origin_image_height)
    
    # Select available ratios based on scale
    if scale <= 2:
        available_ratios = factor_dict[scale] + factor_dict[scale + 1]
    else:
        available_ratios = factor_dict[scale - 1] + factor_dict[scale] + factor_dict[scale + 1]
    
    # Find the factorization that best matches the original aspect ratio
    min_diff = float('inf')
    best_w, best_h = 1, 1
    
    for ratio, w_slice, h_slice in available_ratios:
        log_ratio = math.log(ratio)
        diff = abs(log_ratio - log_origin_ratio)
        if diff < min_diff:
            min_diff = diff
            best_w = w_slice
            best_h = h_slice
    
    return best_w, best_h
def get_patch_nums(origin_image_width: int, origin_image_height: int) -> Tuple[int, int, int, int]:
    """
    Calculate patch numbers for both sliced and abstract image representations.
    
    This function determines the optimal patch distribution for an image that will
    be processed in both sliced format (for detailed analysis) and abstract format
    (for global understanding).
    
    Args:
        origin_image_width (int): Original image width in pixels
        origin_image_height (int): Original image height in pixels
        
    Returns:
        Tuple[int, int, int, int]: 
            - slice_w_num: Number of patches horizontally per slice
            - slice_h_num: Number of patches vertically per slice  
            - abstract_w_num: Number of patches horizontally for full image
            - abstract_h_num: Number of patches vertically for full image
    """
    # Calculate optimal slicing configuration
    best_w, best_h = cal_num_of_slices(origin_image_width, origin_image_height)
    
    # Calculate dimensions of individual slices
    slice_width = origin_image_width // best_w
    slice_height = origin_image_height // best_h
    
    # Get patch numbers for individual slices
    _, _, slice_h_num, slice_w_num = adapt_size(slice_height, slice_width)
    
    # Get patch numbers for the abstract (full) image
    _, _, abstract_h_num, abstract_w_num = adapt_size(origin_image_height, origin_image_width)

    return slice_w_num, slice_h_num, abstract_w_num, abstract_h_num


# =============================================================================
# Image Slicing and Windowing Functions  
# =============================================================================

def slice_image_any_res(image: Image.Image) -> List[Image.Image]:
    """
    Slice an image into optimal sub-regions based on aspect ratio analysis.
    
    This function divides an image into smaller regions that maintain good
    aspect ratios for processing, following the principles of the any-resolution
    image processing pipeline.
    
    Args:
        image (Image.Image): Input PIL image to be sliced
        
    Returns:
        List[Image.Image]: List of image slices as PIL Image objects
    """
    origin_image_width = image.size[0]
    origin_image_height = image.size[1]

    # Calculate optimal slicing configuration
    best_w, best_h = cal_num_of_slices(origin_image_width, origin_image_height)
    
    slices = []
    
    # Extract slices in row-major order
    for j in range(best_h):
        for i in range(best_w):
            # Calculate bounding box for current slice
            left = i * origin_image_width // best_w
            top = j * origin_image_height // best_h
            right = (i + 1) * origin_image_width // best_w
            bottom = (j + 1) * origin_image_height // best_h
            
            box = (left, top, right, bottom)
            region = image.crop(box).convert("RGB")
            slices.append(region)
          
    return slices


def sliding_window(matrix, window_size, stride):
    b,c,height, width = matrix.shape
    
    window_rows = (height - window_size[0]) // stride + 1
    window_cols = (width - window_size[1]) // stride + 1
    windows = []
    for i in range(window_rows):
        windows_col = []
        for j in range(window_cols):
            window = matrix[:,:, i*stride:i*stride+window_size[0],  j*stride:j*stride+window_size[1]]
            windows_col.append(window)
        windows.extend(windows_col)
    return windows

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def slice_image_tensor(image, window_size, stride):
    width, height = image.size
    patches = []
    for y in range(0, height - window_size[1] + 1, stride):
        for x in range(0, width - window_size[0] + 1, stride):
            box = (x, y, x + window_size[0], y + window_size[1])
            patch = image.crop(box)
            patches.append(patch)
    return patches

def slice_image_pil(image, window_size, stride):
    width, height = image.size
    patches = []
    for y in range(0, height - window_size[1] + 1, stride):
        for x in range(0, width - window_size[0] + 1, stride):
            box = (x, y, x + window_size[0], y + window_size[1])
            patch = image.crop(box)
            patches.append(patch)
    return patches

def resize_image(image, target_width):
    width_percent = (target_width / float(image.size[0]))
    target_height = int((float(image.size[1]) * float(width_percent)))
    resized_image = image.resize((target_width, target_height), Image.LANCZOS)
    return resized_image

def process_image_any_res(image, background_color=0):
    image = image.convert("RGB")
    
    slices = slice_image_any_res(image)
    images = [image] + slices

    images = [expand2square(img, background_color) for img in images]
    return images

def process_image_naive(image, background_color=0):
    images = []
    image = expand2square(image, background_color)
    images.append(image)
    
    image = resize_image(image, 1024)
    window_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    stride = 308
    windows = slice_image_pil(image, window_size, stride)
    images.extend(windows)
    return images

# def process_image_naive(image, background_color=0):
#     images = []
#     image = expand2square(image, background_color)
#     image_tensor = TF.to_tensor(image).unsqueeze(0)
#     image_tensor = F.interpolate(image_tensor, size=(644,644), mode='bicubic') # 644 for 4 images 1024 for 9 images

#     window_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
#     stride = 308
#     windows = sliding_window(image_tensor, window_size, stride)
#     image_tensor = F.interpolate(image_tensor, size=(IMAGE_WIDTH,IMAGE_HEIGHT), mode='bicubic')

#     images.append(image_tensor)
#     images.extend(windows)

#     images = torch.cat(images, dim=0)
#     return images


# def process_image(image):
#     origin_image_width  = image.size[0]
#     origin_image_height = image.size[1]

#     image = image.convert("RGB")
    
#     slices = slice_image_any_res(image)
    
#     # 计算resize之后的图片大小
#     resized_height, resized_width, resized_patch_height, resized_patch_width = \
#     adapt_size(origin_image_height,origin_image_width)
    
#     if len(slices) == 1:
#         image = slices[0]
#         image_w = image.size[0]
#         image_h = image.size[1]
#         resized_height, resized_width, resized_patch_height, resized_patch_width = \
#         adapt_size(image_h,image_w)     
        
#         image = ToTensor()(image)
    
#         image = torch.nn.functional.interpolate(
#                 image.unsqueeze(0),
#                 size=(resized_height, resized_width),
#                 mode="bilinear",
#                 align_corners=False,
#                 antialias=True,
#             ).squeeze(0)
#         # 需要mask的patch数
#         num_patches_to_pad = MAX_PATCHES - resized_patch_height*resized_patch_width
#         # raprint("mask: ",num_patches_to_pad)
#         # 切割resize好的图片
#         image = torch_extract_patches(image,PATCH_SIZE, PATCH_SIZE)
#         image = image.reshape([resized_patch_width*resized_patch_height,TOKEN_LENGTH])
#         # 用0补全需要mask的图片部分
#         image = torch.nn.functional.pad(image, [0, 0, 0, num_patches_to_pad]).float()  #torch.Size([196, 768])
#         image = image.reshape(PATCH_NUM_WIDTH, PATCH_NUM_HEIGHT, PATCH_SIZE, PATCH_SIZE, 3).permute(0, 2, 1, 3, 4).reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 3).permute(2, 0 ,1)
#         # print(image)
#         return [image]
    
#     else:
#         images = []
#         resized_patch_widths = []
#         resized_patch_heights = []
#         slices = [image] + slices
#         for image in slices:
#             image = ToTensor()(image)
#             image = torch.nn.functional.interpolate(
#                     image.unsqueeze(0),
#                     size=(resized_height, resized_width),
#                     mode="bilinear",
#                     align_corners=False,
#                     antialias=True,
#                 ).squeeze(0)
#             # 需要mask的patch数
#             num_patches_to_pad = MAX_PATCHES - resized_patch_height*resized_patch_width
#             # raprint("mask: ",num_patches_to_pad)
#             # 切割resize好的图片
#             image = torch_extract_patches(image,PATCH_SIZE, PATCH_SIZE)
#             image = image.reshape([resized_patch_width*resized_patch_height,TOKEN_LENGTH])
#             # 用0补全需要mask的图片部分
#             image = torch.nn.functional.pad(image, [0, 0, 0, num_patches_to_pad]).float()  #torch.Size([196, 768])
#             image = image.reshape(PATCH_NUM_WIDTH, PATCH_NUM_HEIGHT, PATCH_SIZE, PATCH_SIZE, 3).permute(0, 2, 1, 3, 4).reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 3).permute(2, 0 ,1)
            
#             # print(image)
#             images.append(image)
#             resized_patch_widths.append(resized_patch_width)
#             resized_patch_heights.append(resized_patch_height)
            
#         images = torch.stack(images, dim=0)
#         return images