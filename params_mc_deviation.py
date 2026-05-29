import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates

plt.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 15,
    'axes.labelsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 12,
    'figure.titlesize': 15
})

# without regularization parameters
param1_ = [1.036433250037427, 1.049562353210842, 1.049562353210842, 1.06137854105856, 1.0725382837643183, 1.0836980264700764, 1.0935448538501378, 1.102735211046462, 1.102735211046462]
param2_ = [0.966, 0.974, 0.981, 0.981, 0.986, 0.987, 0.987, 0.992, 1.004, 1.004]
param3_ = [0.992, 1.018, 1.018, 1.025, 1.045, 1.072, 1.081, 1.081, 1.140, 1.172]
param4_ = [0.972, 0.983, 0.983, 0.983, 1.018, 1.018, 1.018, 1.036, 1.062, 1.087]
param5_ = [1.087, 0.879, 0.879, 0.937, 0.980, 1.002, 1.005, 1.005, 1.078, 1.126]

# with regularization parameters
param1 = [0.624, 0.625]
param2 = [0.833, 0.834]
param3 = [1.307, 1.308]
param4 = [0.544, 0.545]
param5 = [0.575, 0.576]

# Prepare data for parallel coordinates plot
params_without_reg = [param1_, param2_, param3_, param4_, param5_]
params_with_reg = [param1, param2, param3, param4, param5]

# Calculate means and stds
means_without = [np.mean(p) for p in params_without_reg]
stds_without = [np.std(p) for p in params_without_reg]
means_with = [np.mean(p) for p in params_with_reg]
stds_with = [np.std(p) for p in params_with_reg]

# Create dataframe for parallel_coordinates with mean and std bounds
data = {
    'Param 1': [means_without[0], means_with[0]],
    'Param 2': [means_without[1], means_with[1]],
    'Param 3': [means_without[2], means_with[2]],
    'Param 4': [means_without[3], means_with[3]],
    'Param 5': [means_without[4], means_with[4]],
    'Regularisation': ['Without', 'With']
}

df = pd.DataFrame(data)

# Create figure with parallel coordinates
fig, ax = plt.subplots(figsize=(10, 6))
parallel_coordinates(df, 'Regularisation', ax=ax, color=['steelblue', 'coral'], linewidth=2.5, marker='o', markersize=8)

# Add shaded regions for std
x_positions = np.arange(len(df.columns) - 1)
ax.fill_between(x_positions, 
                 np.array(means_without) - np.array(stds_without),
                 np.array(means_without) + np.array(stds_without),
                 alpha=0.2, color='steelblue', label='Without Regularisation (±std)')
ax.fill_between(x_positions,
                 np.array(means_with) - np.array(stds_with),
                 np.array(means_with) + np.array(stds_with),
                 alpha=0.2, color='coral', label='With Regularisation (±std)')

ax.set_ylabel('Value')
ax.set_title('Parameter Deviation Comparison')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('./plots/parameters_deviation.png', bbox_inches='tight')
plt.show()
