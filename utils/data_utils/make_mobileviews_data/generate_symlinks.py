import os
import glob
from pathlib import Path
from tqdm import tqdm
import argparse

def create_symlinks(source_dir, target_dir, file_pattern="*.jpg"):
    """
    Create symbolic links for all matching files from source directory to target directory.
    
    Args:
        source_dir (str): Root directory to search for files
        target_dir (str): Directory where symlinks will be created
        file_pattern (str): File pattern to match (default: "*.jpg")
    """
    # Create target directory if it doesn't exist
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files recursively
    source_path = Path(source_dir)
    all_files = list(source_path.rglob(file_pattern))
    
    print(f"Found {len(all_files)} {file_pattern} files")
    
    # Create symlinks with progress bar
    for file_path in tqdm(all_files, desc="Creating symlinks"):
        # Create a unique name for the symlink
        relative_path = file_path.relative_to(source_path)
        unique_name = str(relative_path).replace(os.sep, '_')
        link_path = target_path / unique_name
        
        # Create symlink if it doesn't exist
        if not link_path.exists():
            try:
                os.symlink(file_path, link_path)
            except OSError as e:
                print(f"Error creating symlink for {file_path}: {e}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create symlinks for jpg files')
    parser.add_argument('--source_dir', help='Source directory to search for files', default="/mnt/vdb1/hongxin_li/MobileViews/" + ["MobileViews_0-150000", "MobileViews_150001-291197", "MobileViews_300000-400000", "MobileViews_400001-522301"][3])#, required=False)
    parser.add_argument('--target_dir', help='Target directory for symlinks', default="/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/mobileviews")#, required=False)
    parser.add_argument('--pattern', default='*.jpg', help='File pattern to match (default: *.jpg)')
    
    args = parser.parse_args()
    
    # Create symlinks
    print(f"Generate for {args.source_dir}")
    create_symlinks(args.source_dir, args.target_dir, args.pattern)
    
if __name__ == "__main__":
    main()