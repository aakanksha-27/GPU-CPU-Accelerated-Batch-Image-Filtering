# GPU-CPU-Accelerated-Batch-Image-Filtering

## Performance Evaluation (CPU vs GPU)

To demonstrate the benefit of GPU acceleration, we compare batch image
processing performance on CPU and GPU using identical operations
(Gaussian blur + Sobel edge detection).

The experiment processes hundreds of MNIST images in a single execution.

Timing results are recorded in `logs/timing.txt` and show a clear
performance improvement when using GPU-based execution via CUDA-backed
libraries.
