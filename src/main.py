import argparse
import numpy as np
import cv2
from cpu_pipeline import cpu_process_images
from gpu_pipeline import gpu_process_images
import os

def load_images(path, limit):
    images = []
    for fname in sorted(os.listdir(path))[:limit]:
        img = cv2.imread(os.path.join(path, fname), cv2.IMREAD_GRAYSCALE)
        images.append(img.astype(np.float32))
    return np.stack(images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=500)
    args = parser.parse_args()

    images = load_images("data/input", args.num_images)

    cpu_out, cpu_time = cpu_process_images(images)
    gpu_out, gpu_time = gpu_process_images(images)

    with open("logs/timing.txt", "w") as f:
        f.write(f"Images processed: {args.num_images}\n")
        f.write(f"CPU time: {cpu_time:.4f} seconds\n")
        f.write(f"GPU time: {gpu_time:.4f} seconds\n")
        f.write(f"Speedup: {cpu_time / gpu_time:.2f}x\n")

    print("Timing comparison complete.")
