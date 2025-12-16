import cv2
import numpy as np
import time
import os

def cpu_process_images(images):
    start = time.time()

    outputs = []
    for img in images:
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        edges = cv2.Sobel(blurred, cv2.CV_64F, 1, 1, ksize=3)
        outputs.append(edges)

    end = time.time()
    return outputs, end - start
