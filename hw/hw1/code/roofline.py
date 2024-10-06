import matplotlib.pyplot as plt
import numpy as np

# Roofline parameters
peak_flops = 200  # GFLOPS
peak_bandwidth = 30  # GB/s

def roofline(x, peak_flops, peak_bandwidth):
    return np.minimum(peak_flops, x * peak_bandwidth)

plt.figure(figsize=(12, 8))

x = np.logspace(-2, 2, 1000)
plt.loglog(x, roofline(x, peak_flops, peak_bandwidth), 'b-', label='Roofline')

ai = peak_flops / peak_bandwidth
plt.axvline(x=ai, color='r', linestyle='--', label='Peak AI')

# Your measurement data
# Format: (Arithmetic Intensity, Performance in GFLOPS, Label, Failed)
measurements = [
    (0.25, 1.585190, "c1 300M", False),
    (0.25, 3.457980, "c2 300M", False),
    (0.25, 3.779775 , "c3 300M", False),
    (0.25, 0.000001, "c4 300M", True),  # Using a very small value to indicate failure
    (0.25, 3.628275, "c5 300M", False),
    (0.25, 1.672034, "c1 1M", False),
    (0.25, 5.574774, "c2 1M", False),
    (0.25, 8.291145, "c3 1M", False),
    (0.25, 0.010834, "c4 1M", False),
    (0.25, 8.167244, "c5 1M", False),
]

# Plot measurement points
for ai, perf, label, failed in measurements:
    if failed:
        plt.plot(ai, perf, 'kx', markersize=10, label=f"{label} (Failed)")
    else:
        plt.plot(ai, perf, 'ro', markersize=8)
    plt.annotate(label, (ai, perf), xytext=(5, 5), textcoords='offset points')

plt.xlabel('Arithmetic Intensity (FLOPS/Byte)')
plt.ylabel('Performance (GFLOPS)')
plt.title('Roofline Model with Measurements')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)

plt.xlim(0.1, 100)
plt.ylim(0.000001, 300)  # Extended lower limit to show failed execution

plt.savefig('roofline_model.png', dpi=300, bbox_inches='tight')
plt.close()
