import os
import shutil
from tqdm import tqdm

def symlink_images_to_unified_folder(source_directory, destination_directory):
    """
    Creates symlinks for all image files from a source directory (and its subdirectories)
    to a single, unified destination directory.

    Args:
        source_directory (str): The path to the directory containing images
                                (and subdirectories with images).
        destination_directory (str): The path to the directory where all
                                     image symlinks will be created.
    """
    # Ensure the destination directory exists, create it if it doesn't
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
        print(f"Created destination directory: {destination_directory}")
    else:
        print(f"Destination directory already exists: {destination_directory}")

    # Define common image file extensions (can be extended)
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg')

    linked_count = 0
    skipped_count = 0
    existing_count = 0
    duplicate_names = {} # To handle files with the same name

    print(f"\nScanning for images in '{source_directory}'...")
    
    # First pass: count total image files for progress bar
    total_images = 0
    for root, _, files in os.walk(source_directory):
        for file_name in files:
            if file_name.lower().endswith(image_extensions):
                total_images += 1
    
    print(f"Found {total_images} image files to symlink.")
    print(f"Starting to create symlinks from '{source_directory}' to '{destination_directory}'...")

    # Second pass: create symlinks with progress bar
    with tqdm(total=total_images, desc="Creating symlinks", unit="files", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        for root, _, files in os.walk(source_directory):
            for file_name in files:
                # Check if the file is an image based on its extension
                if file_name.lower().endswith(image_extensions):
                    source_file_path = os.path.join(root, file_name)
                    base_name, extension = os.path.splitext(file_name)
                    destination_file_name = file_name
                    destination_file_path = os.path.join(destination_directory, destination_file_name)

                    # Check if symlink already exists
                    if os.path.exists(destination_file_path):
                        existing_count += 1
                        pbar.set_description(f"Skipping existing: {os.path.basename(file_name)[:25]}")
                        pbar.update(1)
                        continue

                    # Handle potential duplicate file names in the destination
                    if destination_file_name in os.listdir(destination_directory):
                        if base_name not in duplicate_names:
                            duplicate_names[base_name] = 1
                        else:
                            duplicate_names[base_name] += 1
                        # Append a counter to the filename to make it unique
                        destination_file_name = f"{base_name}_{duplicate_names[base_name]}{extension}"
                        destination_file_path = os.path.join(destination_directory, destination_file_name)
                        tqdm.write(f"  Warning: Duplicate file name '{file_name}' found. Renaming symlink to '{destination_file_name}'.")

                    try:
                        # Create absolute path for the source to ensure symlink works
                        abs_source_path = os.path.abspath(source_file_path)
                        os.symlink(abs_source_path, destination_file_path)
                        linked_count += 1
                        pbar.set_description(f"Linking: {os.path.basename(file_name)[:30]}")
                        pbar.update(1)
                    except Exception as e:
                        tqdm.write(f"  Error creating symlink for '{source_file_path}': {e}")
                        skipped_count += 1
                        pbar.update(1)

    print(f"\nSymlink creation complete!")
    print(f"Total image symlinks created: {linked_count}")
    print(f"Total existing symlinks skipped: {existing_count}")
    print(f"Total files skipped (due to errors): {skipped_count}")
    if duplicate_names:
        print(f"Symlinks renamed due to duplicates: {len(duplicate_names)}")

# --- How to use this script ---
# IMPORTANT: Replace 'your_source_folder_path' and 'your_destination_folder_path'
# with the actual paths on your system.

# Example Usage:
# source_folder = 'C:/Users/YourUser/Pictures' # Example for Windows
# destination_folder = 'C:/Users/YourUser/UnifiedImages' # Example for Windows

source_folder = '/mnt/jfs/copilot/lhx/ui_data/GUIOdyssey_raw/screenshots' # Example for Linux/macOS
destination_folder = '/mnt/jfs/copilot/lhx/ui_data/GUIOdyssey/' # Example for Linux/macOS
os.makedirs(destination_folder, exist_ok=True)

symlink_images_to_unified_folder(source_folder, destination_folder)
# For demonstration, let's use relative paths or create dummy directories
# You can uncomment and modify these lines for actual use:

# # Create some dummy folders and files for testing
# os.makedirs('test_source/folder1', exist_ok=True)
# os.makedirs('test_source/folder2/subfolder', exist_ok=True)
# with open('test_source/folder1/image1.jpg', 'w') as f: f.write('dummy content')
# with open('test_source/folder1/document.txt', 'w') as f: f.write('dummy content')
# with open('test_source/folder2/image2.png', 'w') as f: f.write('dummy content')
# with open('test_source/folder2/subfolder/image1.jpg', 'w') as f: f.write('dummy content') # Duplicate name

# source_folder_path = 'test_source'
# destination_folder_path = 'unified_images_output'

# symlink_images_to_unified_folder(source_folder_path, destination_folder_path)

# # Clean up dummy folders after testing (optional)
# # shutil.rmtree('test_source')
# # shutil.rmtree('unified_images_output')

