import os, json
import argparse
import time
import multiprocessing
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool, Manager

# Add argument parsing
parser = argparse.ArgumentParser(description='Check UGround data for broken images with multiprocessing')
parser.add_argument('--file', type=str, default="/data/hongxin_li/scaling_exp/UIPro_processed/LlamaFactory_data/UGround1228k_AutoGUI411k.jsonl",
                    help='Path to the JSONL file to process')
parser.add_argument('--workers', type=int, default=16, 
                    help='Number of worker processes to use')
args = parser.parse_args()

# Function to check a single sample
def check_sample(sample):
    img_path = sample['images'][0]
    
    is_bad = False
    if os.path.exists(img_path):
        try:
            img = Image.open(img_path)
        except:
            img = None
            is_bad = True
        
        if img is None:
            is_bad = True
    else:
        is_bad = True
        
    return sample, is_bad

# Function to process a batch of samples
def process_batch(batch_args):
    batch, worker_id = batch_args
    results = []
    
    for sample in batch:
        results.append(check_sample(sample))
    
    return results

if __name__ == "__main__":
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    
    print(f"Processing file: {args.file}")
    print(f"Using {args.workers} worker processes")
    
    # Load data
    with open(args.file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Create manager for process-safe variables
    manager = Manager()
    processed_samples = manager.Value('i', 0)
    
    # Prepare batches
    batch_size = max(1, len(data) // (args.workers * 10))  # Create multiple batches per worker
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append((data[i:i+batch_size], i // batch_size))
    
    # Initialize tracking variables
    start_time = time.time()
    bad_samples = []
    new_samples = []
    
    # Process batches using multiprocessing pool
    print(f"Processing {len(data)} samples in {len(batches)} batches...")
    with Pool(processes=args.workers) as pool:
        results_iterator = pool.imap_unordered(process_batch, batches)
        
        # Use tqdm to show progress
        with tqdm(total=len(data)) as pbar:
            for batch_results in results_iterator:
                for sample, is_bad in batch_results:
                    if is_bad:
                        bad_samples.append(sample)
                    else:
                        new_samples.append(sample)
                    
                    # Update progress bar
                    pbar.update(1)
                    processed_samples.value += 1
                    
                    # Update throughput display occasionally
                    if processed_samples.value % 1000 == 0:
                        elapsed = time.time() - start_time
                        throughput = processed_samples.value / elapsed if elapsed > 0 else 0
                        pbar.set_description(f"Throughput: {throughput:.2f} samples/sec | #Bad: {len(bad_samples)} | #Good: {len(new_samples)}")
    
    # Calculate and display final statistics
    elapsed_time = time.time() - start_time
    throughput = len(data) / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
    print(f"Final throughput: {throughput:.2f} samples/second")
    print(f"Bad samples: {len(bad_samples)}")
    print(f"Good samples: {len(new_samples)}")
    
    # Write results to files
    with open(args.file.replace('.jsonl', '_bad.jsonl'), 'w') as f:
        for x in bad_samples:
            f.write(json.dumps(x) + '\n')
    
    with open(args.file.replace('.jsonl', '_new.jsonl'), 'w') as f:
        for x in new_samples:
            f.write(json.dumps(x) + '\n')
    
    print(f"Results written to: {args.file.replace('.jsonl', '_bad.jsonl')} and {args.file.replace('.jsonl', '_new.jsonl')}")
