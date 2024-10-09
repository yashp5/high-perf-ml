import subprocess

commands = [
    "python c2.py --num_workers 2 --optimizer sgd",
    "python c3.py",
    "python c4.py",
    "python c5.py",
    "python c6.py",
    "python c7.py --use_cuda --num_workers 4 --optimizer sgd",
    "python q3.py"
]

for cmd in commands:
    print(f"Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print("Command completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
    print("\n" + "-"*50 + "\n")
