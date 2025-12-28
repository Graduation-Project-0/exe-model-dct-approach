import os
import numpy as np
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import functools
import cv2  # OpenCV for image saving

# Import generation functions
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_generation import (
    create_two_channel_image, 
    read_binary_file, 
    extract_bigrams, 
    create_bigram_image
)

def process_file(file_info):
    """
    Process a single file and save it.
    Args:
        file_info: tuple (src_path, output_class_dir, filename_base, formats)
    """
    src_path, output_class_dir, filename_base, formats = file_info
    
    try:
        # Generate 2-channel image
        image = create_two_channel_image(src_path)
        
        status = "success"
        
        # Save as .npy
        if 'npy' in formats:
            npy_path = os.path.join(output_class_dir, 'npy', filename_base + '.npy')
            if not os.path.exists(npy_path):
                np.save(npy_path, image)
            else:
                if len(formats) == 1: status = "skipped"

        # Save as .png
        if 'png' in formats:
            # Save Byteplot (Channel 0)
            byteplot_path = os.path.join(output_class_dir, 'byteplot', filename_base + '.png')
            if not os.path.exists(byteplot_path):
                byteplot = (image[:, :, 0] * 255).astype(np.uint8)
                cv2.imwrite(byteplot_path, byteplot)
            
            # Save Bigram-DCT (Channel 1)
            dct_path = os.path.join(output_class_dir, 'dct', filename_base + '.png')
            if not os.path.exists(dct_path):
                dct_img = image[:, :, 1]
                if dct_img.max() > 0:
                    dct_img = (dct_img / dct_img.max() * 255).astype(np.uint8)
                else:
                    dct_img = (dct_img * 255).astype(np.uint8)
                cv2.imwrite(dct_path, dct_img)

            # Save Raw Bigram (before DCT)
            bigram_path = os.path.join(output_class_dir, 'bigram', filename_base + '.png')
            if not os.path.exists(bigram_path):
                byte_data = read_binary_file(src_path)
                bigram_freq = extract_bigrams(byte_data)
                bigram_img = create_bigram_image(bigram_freq, zero_out_0000=True)
                
                # Normalize for visualization
                if bigram_img.max() > 0:
                    bigram_save = (bigram_img / bigram_img.max() * 255).astype(np.uint8)
                else:
                    bigram_save = (bigram_img * 255).astype(np.uint8)
                
                cv2.imwrite(bigram_path, bigram_save)

        return status
    except Exception as e:
        return f"error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Extract malware images to disk")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root data directory")
    parser.add_argument("--output_dir", type=str, default="./data/processed", help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--format", type=str, default="both", choices=["npy", "png", "both"], help="Output format")
    
    args = parser.parse_args()
    
    formats = []
    if args.format == "both":
        formats = ["npy", "png"]
    else:
        formats = [args.format]
    
    classes = ['benign', 'malware']
    subdirs = []
    if 'npy' in formats: subdirs.append('npy')
    if 'png' in formats: subdirs.extend(['byteplot', 'dct', 'bigram'])

    tasks = []
    
    print(f"Scanning files in {args.data_dir}...")
    
    for class_name in classes:
        src_dir = os.path.join(args.data_dir, class_name)
        dst_dir = os.path.join(args.output_dir, class_name)
        
        if not os.path.exists(src_dir):
            print(f"Source directory {src_dir} does not exist.")
            continue
            
        # Create subdirectories
        for sub in subdirs:
            os.makedirs(os.path.join(dst_dir, sub), exist_ok=True)
        
        for filename in os.listdir(src_dir):
            src_path = os.path.join(src_dir, filename)
            if os.path.isfile(src_path) and not filename.endswith('.npy') and not filename.endswith('.png'):
                tasks.append((src_path, dst_dir, filename, formats))
    
    print(f"Found {len(tasks)} files to process.")
    
    if not tasks:
        return

    print(f"Processing with {args.workers} workers...")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(process_file, tasks), total=len(tasks)))
        
    for res in results:
        if res == "success":
            success_count += 1
        elif res == "skipped":
            skip_count += 1
        else:
            error_count += 1
            
    print("\nProcessing Complete.")
    print(f"Success: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Errors:  {error_count}")
    print(f"Output saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
