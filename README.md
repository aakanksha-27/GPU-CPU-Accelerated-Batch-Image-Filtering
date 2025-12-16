# GPU-CPU-Accelerated-Batch-Image-Filtering

## Overview
This project evaluates the performance benefits of GPU acceleration for batch image processing. The same image-processing pipeline is executed on both CPU and GPU, and execution times are compared to quantify speedup.

The project is designed to demonstrate how batching workloads improves GPU utilization and throughput compared to CPU-based execution.

## Dataset
The MNIST handwritten digits dataset is used for experimentation.
The pipeline processes hundreds of small grayscale images in a single run, making it well-suited for evaluating batch GPU performance.

Dataset source:
https://archive-beta.ics.uci.edu/dataset/683/mnist+database+of+handwritten+digits

## Processing Pipeline
Each image batch undergoes the following operations:

- Gaussian blur
- Sobel edge detection

Both CPU and GPU implementations perform identical operations to ensure a fair performance comparison. GPU execution uses CUDA-backed libraries to ensure actual GPU computation.

## How to Run
This project was executed in a GPU-enabled environment (Google Colab).

Steps:
git clone https://github.com/aakanksha-27/GPU-CPU-Accelerated-Batch-Image-Filtering

cd GPU-CPU-Accelerated-Batch-Image-Filtering

chmod +x run.sh

./run.sh

## Performance Evaluation (CPU vs GPU)

To demonstrate the benefit of GPU acceleration, we compare batch image processing performance on CPU and GPU using identical operations (Gaussian blur + Sobel edge detection).

The experiment processes hundreds of MNIST images in a single execution.

Timing results are recorded in `logs/timing.txt` and show a clear performance improvement when using GPU-based execution via CUDA-backed libraries.

## Output / Proof of Execution
This project focuses on performance benchmarking rather than visual output.
The primary output is execution timing (CPU vs GPU), recorded in logs/timing.txt.
These logs demonstrate that GPU execution processes large batches of images significantly faster than CPU execution.

## Design Decisions and Lessons Learned

- Batching is critical for effective GPU utilization; single-image execution does not adequately demonstrate GPU advantages.
- GPU memory transfers must be amortized across large batches to achieve meaningful speedups.
- Performance benchmarking is a valid and practical GPU workload, even without producing visual output.

This project reinforced the importance of workload design when evaluating GPU acceleration and highlighted the trade-offs between computation and memory transfer overheads.
