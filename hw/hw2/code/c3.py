import subprocess
import re

def run_training(num_workers):
    cmd = f"python c2.py --num_workers {num_workers}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout
    data_loading_times = re.findall(r"Data Loading Time: (\d+\.\d+)s", output)
    if data_loading_times:
        return float(data_loading_times[-1])  # Return the last (most recent) data loading time
    return None

def main():
    worker_counts = []
    data_loading_times = []

    # Start with 0 workers and increase by 4 until data loading time does not improve
    num_workers = 0
    prev_time = float('inf')

    while True:
        print(f"Running with {num_workers} workers...")
        time = run_training(num_workers)

        if time is not None:
            worker_counts.append(num_workers)
            data_loading_times.append(time)
            print(f"Data loading time for {num_workers} workers: {time}s")

            # Stop if the data loading time stops decreasing
            if time >= prev_time:
                break

            prev_time = time
        else:
            print(f"Failed to get data loading time for {num_workers} workers")

        num_workers += 4

    # Print the results
    print("\nSummary of Results:")
    for workers, time in zip(worker_counts, data_loading_times):
        print(f"Number of Workers: {workers}, Data Loading Time: {time:.2f}s")

    # Find the best number of workers
    best_workers = worker_counts[data_loading_times.index(min(data_loading_times))]
    print(f"\nThe best number of workers for runtime performance is: {best_workers}")

if __name__ == "__main__":
    main()
