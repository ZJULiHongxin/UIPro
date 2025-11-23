#!/bin/bash

# Script: Process four MobileViews dataset folders sequentially and generate symbolic links
# Usage: bash run_generate_symlinks.sh

# Set base paths
BASE_SOURCE_DIR="/mnt/vdb1/hongxin_li/MobileViews"
TARGET_DIR="/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/mobileviews"
PYTHON_SCRIPT="utils/data_utils/make_mobileviews_data/generate_symlinks.py"

# List of four dataset folders
DATASETS=(
    "MobileViews_0-150000"
    "MobileViews_150001-291197"
    "MobileViews_300000-400000"
    "MobileViews_400001-522301"
)

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Check if base source directory exists
if [ ! -d "$BASE_SOURCE_DIR" ]; then
    echo "Error: Source directory does not exist: $BASE_SOURCE_DIR"
    exit 1
fi

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Loop through each dataset folder
total=${#DATASETS[@]}
current=0

for dataset in "${DATASETS[@]}"; do
    current=$((current + 1))
    source_dir="${BASE_SOURCE_DIR}/${dataset}"
    
    echo "=========================================="
    echo "Processing dataset [$current/$total]: $dataset"
    echo "Source directory: $source_dir"
    echo "Target directory: $TARGET_DIR"
    echo "=========================================="
    
    # Check if source directory exists
    if [ ! -d "$source_dir" ]; then
        echo "Warning: Source directory does not exist, skipping: $source_dir"
        continue
    fi
    
    # Run Python script
    python3 "$PYTHON_SCRIPT" \
        --source_dir "$source_dir" \
        --target_dir "$TARGET_DIR" \
        --pattern "*.jpg"
    
    # Check exit status of previous command
    if [ $? -ne 0 ]; then
        echo "Error: Error occurred while processing $dataset"
        echo "Continue processing next dataset? (y/n)"
        read -r response
        if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
            echo "User cancelled, exiting script"
            exit 1
        fi
    else
        echo "Successfully completed: $dataset"
    fi
    
    echo ""
done

echo "=========================================="
echo "All datasets processed successfully!"
echo "=========================================="

