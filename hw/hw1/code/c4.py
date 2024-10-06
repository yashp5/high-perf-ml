import numpy as np
import time
import argparse
import sys

def calculate_mean_time_taken(exec_times, reps):
    """
    Calculate the mean of the second half of the execution times.
    """
    start = reps // 2
    relevant_times = exec_times[start:]
    return np.mean(relevant_times)

def calculate_bandwidth(N, mean_time_taken):
    """
    Calculate Bandwidth in GB/sec.
    """
    data_moved_bytes = 2.0 * N * np.float32().nbytes  # 2 arrays of N float32 elements
    data_moved_gb = data_moved_bytes / 1e9
    bandwidth = data_moved_gb / mean_time_taken
    return bandwidth

def calculate_throughput(N, mean_time_taken):
    """
    Calculate Throughput in GFLOP/sec.
    """
    flops = 2.0 * N  # N multiplications and N additions
    throughput = (flops / mean_time_taken) / 1e9
    return throughput

def dp(N, A, B):
    R = 0.0;
    for j in range(0,N):
        R += A[j]*B[j]
    return R

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Dot Product Performance Measurement")
    parser.add_argument('N', type=int, help='Size of the vectors (positive integer)')
    parser.add_argument('reps', type=int, help='Number of repetitions (positive integer)')
    args = parser.parse_args()

    N = args.N
    reps = args.reps

    # Validate inputs
    if N <= 0:
        print("Error: N must be a positive integer", file=sys.stderr)
        sys.exit(1)
    if reps <= 0:
        print("Error: reps must be a positive integer", file=sys.stderr)
        sys.exit(1)

    try:
        # Initialize arrays with dtype float32
        pA = np.ones(N, dtype=np.float32)
        pB = np.ones(N, dtype=np.float32)
    except MemoryError:
        print("Error: Memory allocation failed", file=sys.stderr)
        sys.exit(1)

    exec_times = []
    result = 0.0

    for i in range(reps):
        start_time = time.perf_counter()
        result = dp(N, pA, pB)
        end_time = time.perf_counter()

        time_taken = end_time - start_time
        exec_times.append(time_taken)

    # Validate the result
    expected = float(N)
    if not np.isclose(result, expected):
        print(f"Error: Dot product result incorrect (expected {expected}, got {result})", file=sys.stderr)
        sys.exit(1)

    # Calculate performance metrics
    exec_times_array = np.array(exec_times)
    mean_time_taken = calculate_mean_time_taken(exec_times_array, reps)
    bandwidth = calculate_bandwidth(N, mean_time_taken)
    throughput = calculate_throughput(N, mean_time_taken)

    # Print the results
    print(f"N: {N} <T>: {mean_time_taken:.6f} sec B: {bandwidth:.6f} GB/sec F: {throughput:.6f} GFLOP/sec")

if __name__ == "__main__":
    main()
