import subprocess
import re
import numpy as np

def run_training(use_gpu, num_workers):
    cmd = f"python c2.py --num_workers {num_workers} {'--use_cuda' if use_gpu else ''}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout
    # Extract total epoch times from the output
    epoch_times = re.findall(r"Total Epoch Time: (\d+\.\d+)s", output)
    return [float(time) for time in epoch_times] if epoch_times else None

def print_comparison(gpu_times, cpu_times):
    print("\nEpoch-wise comparison:")
    print("Epoch | GPU Time (s) | CPU Time (s) | Speedup")
    print("-" * 50)
    for i, (gpu_time, cpu_time) in enumerate(zip(gpu_times, cpu_times), 1):
        speedup = cpu_time / gpu_time
        print(f"{i:5d} | {gpu_time:12.2f} | {cpu_time:12.2f} | {speedup:7.2f}x")

def main():
    # Replace this with the optimal number of workers you found
    optimal_workers = 4

    print(f"Running training with GPU (using {optimal_workers} workers)...")
    gpu_times = run_training(use_gpu=True, num_workers=optimal_workers)

    print(f"Running training with CPU (using {optimal_workers} workers)...")
    cpu_times = run_training(use_gpu=False, num_workers=optimal_workers)

    if gpu_times and cpu_times:
        avg_gpu_time = np.mean(gpu_times)
        avg_cpu_time = np.mean(cpu_times)

        print(f"\nResults over {len(gpu_times)} epochs:")
        print(f"Average GPU training time per epoch: {avg_gpu_time:.2f} seconds")
        print(f"Average CPU training time per epoch: {avg_cpu_time:.2f} seconds")
        print(f"Overall speedup factor: {avg_cpu_time / avg_gpu_time:.2f}x")

        print_comparison(gpu_times, cpu_times)
    else:
        print("Failed to get timing results for one or both runs.")

if __name__ == "__main__":
    main()
