import cupy as cp
import cupyx.scipy.ndimage as ndi
import time

def gpu_process_images(images):
    # Move batch to GPU
    gpu_imgs = cp.asarray(images)

    # Warm-up (important for fair timing)
    _ = ndi.gaussian_filter(gpu_imgs, sigma=1)
    cp.cuda.Stream.null.synchronize()

    start = time.time()

    blurred = ndi.gaussian_filter(gpu_imgs, sigma=1)
    sobel_x = ndi.sobel(blurred, axis=1)
    sobel_y = ndi.sobel(blurred, axis=2)
    edges = cp.sqrt(sobel_x**2 + sobel_y**2)

    cp.cuda.Stream.null.synchronize()
    end = time.time()

    return edges, end - start
