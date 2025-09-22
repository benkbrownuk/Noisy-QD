# Noisy Quadratic Dataset Visualization (Colab)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Function to generate noisy quadratic data
def generate_noisy_quadratic(n=100, a=0.5, b=0, c=2, noise=2.0):
    x = np.linspace(-5, 5, n)
    y_true = a * x**2 + b * x + c
    y_noisy = y_true + np.random.uniform(-noise, noise, size=n)
    return x, y_true, y_noisy

# Piecewise constant fit (binning)
def piecewise_constant_fit(x, y, bins=5):
    bin_edges = np.linspace(x.min(), x.max(), bins+1)
    bin_means = np.zeros(bins)
    for i in range(bins):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i+1])
        if np.any(mask):
            bin_means[i] = y[mask].mean()
        else:
            bin_means[i] = 0
    # For step plot
    step_x = np.repeat(bin_edges, 2)[1:-1]
    step_y = np.repeat(bin_means, 2)
    return step_x, step_y, bin_edges, bin_means

# Plotting function
def plot_noisy_quadratic_grid(rows=3, bins=5):
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(rows, 2, width_ratios=[2, 1], wspace=0.25, hspace=0.35)
    for row in range(rows):
        x, y_true, y_noisy = generate_noisy_quadratic()
        step_x, step_y, bin_edges, bin_means = piecewise_constant_fit(x, y_noisy, bins)
        # Left: Data, true curve, fit
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.scatter(x, y_noisy, color='royalblue', s=10, label='Noisy Data', alpha=0.7)
        ax1.plot(x, y_true, color='seagreen', lw=2, label='True Quadratic')
        ax1.plot(step_x, step_y, color='purple', lw=2, label='Piecewise Fit')
        ax1.set_ylim(-2.5, 14)
        ax1.set_xlim(-5, 5)
        ax1.set_title('Noisy Data, True Curve, Fit')
        ax1.legend(loc='upper left', fontsize=8)
        # Right: Residuals
        ax2 = fig.add_subplot(gs[row, 1])
        # Assign each x to a bin for residuals
        bin_idx = np.digitize(x, bin_edges) - 1
        bin_idx[bin_idx == bins] = bins - 1  # rightmost edge
        residuals = y_noisy - bin_means[bin_idx]
        ax2.scatter(x, residuals, color='crimson', s=10, label='Residuals', alpha=0.7)
        ax2.axhline(0, color='gray', lw=1, ls='--')
        ax2.plot(step_x, np.zeros_like(step_x), color='purple', lw=2)
        ax2.set_ylim(-6, 4)
        ax2.set_xlim(-5, 5)
        ax2.set_title('Residuals')
    plt.suptitle('Points sampled from a quadratic curve with random noise.', fontsize=16)
    plt.show()

# Run the plot
test_runs = 1
for _ in range(test_runs):
    plot_noisy_quadratic_grid(rows=3, bins=5)
