import subprocess
import matplotlib.pyplot as plt
import re

def run_training(num_workers):
    cmd = f"python c2.py --num_workers {num_workers}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout

    # Extract data loading and training times from the output
    data_loading_times = re.findall(r"Data Loading Time: (\d+\.\d+)s", output)
    training_times = re.findall(r"Training Time: (\d+\.\d+)s", output)

    if data_loading_times and training_times:
        return float(data_loading_times[-1]), float(training_times[-1])
    return None, None

def plot_comparison(single_worker, optimal_workers):
    labels = ['1 Worker', f'{optimal_workers} Workers']
    data_loading_times = [single_worker[0], optimal_workers[0]]
    training_times = [single_worker[1], optimal_workers[1]]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar([i - width/2 for i in x], data_loading_times, width, label='Data Loading Time')
    rects2 = ax.bar([i + width/2 for i in x], training_times, width, label='Training Time')

    ax.set_ylabel('Time (seconds)')
    ax.set_title('Comparison of Data Loading and Training Times')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.savefig('time_comparison.png')

def main():
    # Assuming the optimal number of workers is 8 (you should replace this with the actual number you found)
    optimal_workers = 4

    print("Running with 1 worker...")
    single_worker_results = run_training(1)

    print(f"Running with {optimal_workers} workers...")
    optimal_workers_results = run_training(optimal_workers)

    if None not in single_worker_results and None not in optimal_workers_results:
        print(f"1 Worker - Data Loading Time: {single_worker_results[0]:.2f}s, Training Time: {single_worker_results[1]:.2f}s")
        print(f"{optimal_workers} Workers - Data Loading Time: {optimal_workers_results[0]:.2f}s, Training Time: {optimal_workers_results[1]:.2f}s")

        plot_comparison(single_worker_results, optimal_workers_results)
    else:
        print("Failed to get timing results for one or both runs.")

if __name__ == "__main__":
    main()
