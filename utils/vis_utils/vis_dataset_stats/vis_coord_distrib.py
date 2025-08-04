import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

def parse_bbox_to_points(bbox):
    """
    Convert a bounding box to its four corner points.
    Format (x1, y1, x2, y2) -> four points (x1,y1), (x2,y1), (x1,y2), (x2,y2)
    """
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        return [(x1+x2)//2, (y1+y2)//2]
    else:
        return None

def check_coord_data_type(x, scale):
    return 0 <= x < scale and isinstance(x, int)

def parse_coord_from_response(response, scale):
    """
    Extract coordinates from a response string.
    Response can be a single point (x,y) or a bounding box (x1,y1,x2,y2).
    """
    try:
        # Try to parse as a tuple/list
        coords_str = response[response.rfind('('):response.rfind(')')+1]
        coords = eval(coords_str)
        
        
        if all(check_coord_data_type(coord, scale) for coord in coords):
            if len(coords) == 2:  # Single point
                return coords
            elif len(coords) == 4:  # Bounding box
                return parse_bbox_to_points(coords)
        return None
    except:
        return None

def main():
    parser = argparse.ArgumentParser(description='Visualize coordinate distribution')
    parser.add_argument('--data_path', type=str, required=False, default="/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/UIPro_processed/UIPro_18643k.json", help='Path to the dataset file')
    parser.add_argument('--output_dir', type=str, default=os.path.join(os.path.dirname(__file__), './coord_distrib'), help='Output directory for visualization')
    parser.add_argument('--bins', type=int, default=100, help='Number of bins for the heatmap')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize count matrix for the heatmap
    heatmap = np.zeros((args.bins, args.bins)) # (y, x)
    
    # Load the dataset
    print(f"Loading dataset from {args.data_path}...")
    with open(args.data_path, 'r') as f:
        dataset = json.load(f)
    
    # Process each sample in the dataset
    print(f"Processing {len(dataset)} samples...")
    
    total_points = 0
    for sample in tqdm(dataset):
        # Extract conversations
        if 'conversations' not in sample:
            continue
        
        convs = sample['conversations']
        points = []
        for i in range(1, len(convs), 2):
            # Look for GPT responses which might contain coordinates
            if 'list' in convs[i-1]['value']:
                for line in convs[i]['value'].split('\n'):
                    coords = parse_coord_from_response(line, args.bins)
                    points.append(coords)
            elif convs[i]['value'].startswith('(') and convs[i]['value'].endswith(')'):
                coords = parse_coord_from_response(convs[i]['value'], args.bins)
                points.append(coords)
        total_points += len(points)
        for point in points:
            if point:
                if 0 <= point[0] <= args.bins and 0 <= point[1] <= args.bins and isinstance(point[0], int) and isinstance(point[1], int):
                    heatmap[point[1], point[0]] += 1
    # Create the heatmap visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(heatmap, cmap='viridis', origin='lower')
    plt.colorbar(label='Frequency')
    plt.title(f'Coordinate Distribution Heatmap\nTotal Points: {total_points}')
    plt.xlabel('X Coordinate (Normalized)')
    plt.ylabel('Y Coordinate (Normalized)')
    
    # Save the heatmap
    output_path = os.path.join(args.output_dir, os.path.basename(args.data_path).split('.')[0] + '_coord_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_path}")
    
    # Also save the raw heatmap data
    np.save(os.path.join(args.output_dir, os.path.basename(args.data_path).split('.')[0] + '_heatmap_data.npy'), heatmap)
    
    # Print some statistics
    print(f"Total points processed: {total_points}")
    print(f"Maximum frequency in a bin: {np.max(heatmap)}")
    print(f"Average frequency per bin: {np.sum(heatmap) / (args.bins * args.bins)}")

if __name__ == "__main__":
    main()
