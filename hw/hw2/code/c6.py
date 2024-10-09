import subprocess
import re
import pandas as pd

def run_training(optimizer, optimal_workers):
    cmd = f"python c2.py --use_cuda --num_workers {optimal_workers} --optimizer {optimizer}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout

def parse_output(output):
    epochs = re.findall(r"Epoch \[(\d+)/5\]", output)
    training_times = re.findall(r"Training Time: (\d+\.\d+)s", output)
    losses = re.findall(r"Loss: (\d+\.\d+)", output)
    accuracies = re.findall(r"Accuracy: (\d+\.\d+)%", output)
    return list(zip(epochs, training_times, losses, accuracies))

def print_results(df):
    print("\nDetailed Results:")
    for optimizer in df['Optimizer'].unique():
        print(f"\n{optimizer}:")
        data = df[df['Optimizer'] == optimizer]
        print("Epoch | Training Time (s) | Loss | Accuracy (%)")
        print("-" * 50)
        for _, row in data.iterrows():
            print(f"{row['Epoch']:5d} | {row['Training Time']:16.2f} | {row['Loss']:4.4f} | {row['Accuracy']:12.2f}")

def main():
    optimal_workers = 4
    optimizers = ['sgd', 'sgd_nesterov', 'adagrad', 'adadelta', 'adam']
    results = []

    for optimizer in optimizers:
        print(f"Running training with {optimizer.upper()}...")
        output = run_training(optimizer, optimal_workers)
        parsed_results = parse_output(output)
        for epoch, time, loss, accuracy in parsed_results:
            results.append({
                'Optimizer': optimizer.upper(),
                'Epoch': int(epoch),
                'Training Time': float(time),
                'Loss': float(loss),
                'Accuracy': float(accuracy)
            })

    df = pd.DataFrame(results)

    # Print summary
    summary = df.groupby('Optimizer').agg({
        'Training Time': 'mean',
        'Loss': 'mean',
        'Accuracy': 'mean'
    }).round(4)

    print("\nSummary (averaged over 5 epochs):")
    print(summary)

    # Print detailed results
    print_results(df)

if __name__ == "__main__":
    main()
