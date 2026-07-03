import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os

# Hi-DPI output in Jupyter
try:
    get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
except Exception:
    pass

mpl.rcParams.update({
    # Sizes (your chosen 15 everywhere)
    'font.size': 15,
    'axes.titlesize': 15,
    'axes.labelsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'figure.titlesize': 15,

    # Figure + save defaults
    'figure.figsize': (6.0, 4.0),      # tweak as needed across the thesis
    'figure.dpi': 120,                 # higher dpi for notebooks
    'savefig.dpi': 300,                # print-friendly exports
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,

    # Lines, markers, grids
    'lines.linewidth': 2.0,
    'lines.markersize': 5.5,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,

    # Spines and ticks
    'axes.spines.top': True,
    'axes.spines.right': True,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 4,
    'ytick.major.size': 4,

    # Legends
    'legend.frameon': False,
    'legend.handlelength': 2.0,

    # Colormap (keeps things consistent if imshow/contours used)
    'image.cmap': 'viridis',

    # Layout
    'figure.autolayout': True,  # similar to calling plt.tight_layout() automatically
})

# Read the Excel file
df = pd.read_excel('./start_vs_lambda2_run3/foldername.xlsx')

# Create the plot with subplot
fig, ax = plt.subplots()

# True value
true_val = 0.5 * 0.0031830989

# Plot each lambda column with different colors and styles
# colors = ['blue', 'red', 'green', 'orange', 'purple']

colors = ['blue', 'orange', 'green', 'red', 'purple']
labels = [
    r'$10^{-3}$',
    r'$2×10^{-3}$', 
    r'$10^{-2}$',
    r'$10^{-4}$',
    r'$-10^{-3}$'
]

# Main curves
lines = []
for i, col in enumerate(df.columns[1:]):  # Skip 'Iteration' column
    line, = ax.plot(df['Iteration'], df[col], color=colors[i], label=labels[i], linewidth=1.5)
    lines.append(line)

# True value (main) as dashed red
true_line, = ax.plot(true_val * np.ones_like(df['Iteration']), 'r--')

# Axis labels
ax.set_xlabel('Iterations')
ax.set_ylabel(r'Fitting parameter $\lambda$')

# Legend for main plot
legend_labels = labels + [r'True value ($0.005/\pi$)']
ax.legend(lines + [true_line], legend_labels, loc='upper right', fontsize=15)

# --- Inset: last quarter of iterations, placed outside on the right ---
n = len(df['Iteration'])
start = n * 3 // 4
x_in = df['Iteration'].iloc[start:]

axins = inset_axes(
    ax,
    width="42%", height="42%",
    loc="upper left",
    bbox_to_anchor=(1.15, 0.0, 1.0, 1.0),  # outside right
    bbox_transform=ax.transAxes,
    borderpad=0.0
)

# Inset curves
inset_lines = []
for i, col in enumerate(df.columns[1:]):
    line, = axins.plot(x_in, df[col].iloc[start:], color=colors[i])
    inset_lines.append(line)
true_line_inset, = axins.plot(x_in, true_val * np.ones_like(x_in), 'r--')

# Inset limits with padding (include true value)
all_data_inset = np.concatenate([
    *[df[col].iloc[start:].values for col in df.columns[1:]],
    [true_val]
])
ymin, ymax = np.min(all_data_inset), np.max(all_data_inset)
pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
axins.set_ylim(ymin - pad, ymax + pad)
axins.set_xlim(x_in.iloc[0], x_in.iloc[-1])
axins.tick_params(axis='both', which='both', labelsize=12)

# Save the plot
plt.savefig('./start_vs_lambda2_run3/read_excel.png', dpi=300, bbox_inches='tight')

# Print some statistics
print("\nFinal values:")
for i, col in enumerate(df.columns[1:]):
    final_val = df[col].iloc[-1]
    print(f"{labels[i]}: {final_val:.6f}")

    