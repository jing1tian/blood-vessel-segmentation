import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path/to/dataset")
    parser.add_argument("-o", "--output", help="output dir")
    args = parser.parse_args()
    
    groups = [group.split("/")[-1] for group in glob(os.path.join(args.path, "train/*"))]
    for group in groups:
        label_paths = sorted(glob(os.path.join(args.path, "train", group, "labels", "*.tif")))
        slice_paths = [path.replace("/labels/", "/images/").replace("kidney_3_dense", "kidney_3_sparse") for path in label_paths]
        
        h, w = cv2.imread(slice_paths[0], cv2.IMREAD_UNCHANGED).shape
        volume = np.memmap(os.path.join(args.output, f"{group}.mmap"), dtype=np.uint16, shape=(len(slice_paths), h, w), mode="w+")
        volume_mask = np.memmap(os.path.join(args.output, f"{group}_mask.mmap"), dtype=np.uint8, shape=(len(slice_paths), h, w), mode="w+")
        for i, (slice_path, label_path) in tqdm(enumerate(zip(slice_paths, label_paths)), total=len(slice_paths)):
            slice = cv2.imread(slice_path, cv2.IMREAD_UNCHANGED)
            label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            volume[i] = slice
            volume_mask[i] = label
        volume.flush()
        volume_mask.flush()
        
    # merge kidney_3_dense and kidney_3_sparse
    dense_label_paths = sorted(glob(os.path.join(args.path, "train", "kidney_3_dense", "labels", "*.tif")))
    dense_ids = [s.split("/")[-1][:-4] for s in dense_label_paths]
    slice_paths = sorted(glob(os.path.join(args.path, "train", "kidney_3_sparse", "images", "*.tif")))
    label_paths = [s.replace("/images/", "/labels/") for s in slice_paths]
    label_paths = [s.replace("kidney_3_sparse", "kidney_3_dense") if s.split("/")[-1][:-4] in dense_ids else s for s in label_paths]
    
    h, w = cv2.imread(slice_paths[0], cv2.IMREAD_UNCHANGED).shape
    volume = np.memmap(os.path.join(args.output, f"kidney_3.mmap"), dtype=np.uint16, shape=(len(slice_paths), h, w), mode="w+")
    volume_mask = np.memmap(os.path.join(args.output, f"kidney_3_mask.mmap"), dtype=np.uint8, shape=(len(slice_paths), h, w), mode="w+")
    for i, (slice_path, label_path) in tqdm(enumerate(zip(slice_paths, label_paths)), total=len(slice_paths)):
        slice = cv2.imread(slice_path, cv2.IMREAD_UNCHANGED)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        volume[i] = slice
        volume_mask[i] = label
    volume.flush()
    volume_mask.flush()