# Archive

This folder contains legacy scripts, exploratory notebooks, plotting utilities, and saved figures from the inverse Physics-Informed Neural Network (PINN) experiments on the one-dimensional Burgers equation.

The archived files are kept for transparency and for readers who want to understand how the experiments evolved. They are **not** the recommended entry point for reproducing the cleaned workflow. Some files contain hard-coded local paths, notebook-specific outputs, old naming conventions, duplicate experiments, or partially superseded code.

For the cleaned implementation, start from:

```text
Inverse_Labelled_Burgers_Equation.py
```

## Main experiment scripts

| File | Purpose | Notes |
|---|---|---|
| `Inverse_Labelled_Burgers_Equation.py` | Main cleaned Python script for the inverse labelled Burgers PINN experiment. It trains a PINN using boundary/initial-condition data, labelled interior data, and PDE residual collocation points to recover `lambda_2`. | Best entry point among the Python scripts. Keep outside the archive if this is the maintained version. |
| `#labelled_vs_lambda` | Legacy Python script for studying how the number of labelled interior points affects inverse PINN recovery of `lambda_2`. | Rename to `labelled_vs_lambda.py`. Avoid `#` in filenames because it can be confused with comments or shell syntax. |
| `start_vs_lambda` | Legacy Python script for running initial-lambda sensitivity experiments. It tests different starting values for the trainable `lambda_2` parameter. | Rename to `start_vs_lambda.py` if kept as executable code. |
| `depth_vs_lambda.py` | Legacy Python script for testing the effect of network depth on inverse PINN convergence and `lambda_2` recovery. | Runs one selected depth case at a time and saves fold-wise results. |
| `width_vs_lambda.py` | Legacy Python script for testing the effect of network width on inverse PINN convergence and `lambda_2` recovery. | Runs one selected width case at a time and saves fold-wise results. |

## Plotting and post-processing scripts

| File | Purpose | Notes |
|---|---|---|
| `general_results_plotting.py` | Reads a single `results.xlsx` file and plots fold-averaged training loss, test loss, relative L2 error, and recovered `lambda_2`. | Useful for visualising one completed run. |
| `N_l_plotting.py` | Compares multiple labelled-data sensitivity runs. It reads `results.xlsx` files from folders such as `N_l_10`, `N_l_100`, `N_l_1000`, and `N_l_10000`. | Use after running the labelled-data experiments. |
| `start_lambda_plotting.py` | Compares multiple initial-lambda experiments. It reads the results from different `lambda_*` folders and plots the effect of the initial guess on convergence. | Post-processing companion to `start_vs_lambda`. |
| `depth_plotting.py` | Compares different network-depth experiments by reading results from `depth_*` folders. | Post-processing companion to `depth_vs_lambda.py`. |
| `width_plotting.py` | Compares different network-width experiments by reading results from `width_*` folders. | Post-processing companion to `width_vs_lambda.py`. |
| `width_plotting` | Extensionless/empty placeholder file. | Not useful unless it contains code locally. Consider deleting or replacing with `width_plotting.py`. |
| `plot_excel_start.py` | Legacy plotting utility that reads an Excel file from an initial-lambda run and plots several lambda trajectories with an inset. | Superseded by `start_lambda_plotting.py`, but useful as an older plotting reference. |
| `params_mc_deviation.py` | Post-processing script for analysing Monte Carlo variation of fitted parameters. It contains hard-coded parameter arrays, computes mean/std values, and produces parameter-deviation plots. | Belongs to parameter-stability analysis rather than the main Burgers PINN workflow. |
| `training_callback.py` | Experimental callback class for logging total loss, loss components, `lambda_2`, and relative L2 error during optimisation. | Useful if the main PINN class is refactored to use a cleaner callback pattern. |

## Legacy notebooks

| File | Purpose | Notes |
|---|---|---|
| `Inverse_Labelled_Burgers_Equation.ipynb` | Notebook version of the inverse labelled Burgers PINN. It contains the full workflow: data preparation, labelled/boundary/collocation sampling, PINN definition, L-BFGS-B optimisation, and diagnostic plots. | Useful for development history and interactive inspection. The `.py` version is cleaner for GitHub. |
| `~Inverse_Labelled_Burgers_Equation.ipynb` | Temporary or older notebook version of the inverse labelled PINN experiment. | Archive only. The `~` prefix suggests a temporary/autosave-style file; do not present it as a main result. |
| `Labelled_Burgers_Equation.ipynb` | Earlier labelled-data Burgers PINN notebook using labelled interior points and collocation points. | Useful for understanding the transition from labelled PINN experiments to the inverse labelled workflow. |
| `Burgers_Equation.ipynb` | Early Burgers-equation PINN baseline notebook. | Useful for baseline development history. Likely superseded by the inverse labelled versions. |
| `Burgers_Equation-Self_Adaptive.ipynb` | Experimental self-adaptive Burgers PINN notebook using adaptive loss/gradient ideas. | Archive only unless the self-adaptive method is revived. |
| `myPINN_inverse_problem.ipynb` | Early Colab-style inverse PINN notebook. It includes Google Drive paths and early experiments with trainable PDE parameters. | Historical starting point; not expected to run unchanged. |
| `myPINN_inverse_problem_ADAM.ipynb` | Early inverse PINN version using ADAM-style optimisation and validation-loss tracking. | Useful for comparing the development path before the L-BFGS-B workflow. |
| `myPINN_inverse_problem_LBFGS.ipynb` | Early inverse PINN version using L-BFGS-B optimisation. | Useful for understanding how the later SciPy L-BFGS-B implementation evolved. |
| `width_vs_lambda2.ipynb` | Notebook for studying how hidden-layer width affects recovered `lambda_2`. | Superseded by the cleaner `width_vs_lambda.py` and `width_plotting.py` scripts. |
| `depth_vs_lambda2.ipynb` | Notebook for studying how hidden-layer depth affects recovered `lambda_2`. | Superseded by the cleaner `depth_vs_lambda.py` and `depth_plotting.py` scripts. |
| `start_vs_lambda2.ipynb` | Notebook for studying sensitivity to different initial values of `lambda_2`. | Superseded by `start_vs_lambda` and `start_lambda_plotting.py`. |
| `#labelled_vs_lambda2.ipynb` | Notebook for studying how the number of labelled data points affects recovery of `lambda_2`. | Rename without `#` if reused. Superseded by the cleaner labelled-data scripts. |
| `seed_vs_lambda2.ipynb` | Notebook listed as a seed-sensitivity experiment. | Check the internal content before publishing; one uploaded notebook with this size contained labelled-data sensitivity code rather than a clear seed sweep. |
| `variance of parameters.ipynb` | Small notebook for computing variance of fitted parameters from stored parameter values. | Separate from the Burgers PINN workflow; more related to parameter-stability/Monte Carlo analysis. |

## Saved figures and outputs

| File | Purpose | Notes |
|---|---|---|
| `collocation_points_Burgers.png` | Visualises the training data layout: collocation points inside the domain, boundary/initial-condition points, and labelled interior points. | Useful for README or method explanation. |
| `Burgers.png` | Saved visualisation of the Burgers-equation reference solution or PINN solution field. | Keep as a historical output figure. |
| `loss.png` | Legacy training-loss history from an old run. | Diagnostic output; not a maintained result. |
| `error history.png` | Legacy relative-error history from an old run. | Diagnostic output; not a maintained result. |
| `w0.png` | Legacy plot of a tracked neural-network weight during training. | Useful for debugging/history, not central to the final method. |
| `parameters_deviation.png` | Figure showing deviation/variation of fitted parameters across runs. | Output from parameter-deviation or Monte Carlo analysis. |
| `parameter_parallel_coordinates.png` | Parallel-coordinate style visualisation of fitted parameters across runs or regularisation settings. | Output from `params_mc_deviation.py`. |

## Important caveats

- Several archived files contain hard-coded Windows paths or Colab/Google Drive paths. Update paths before running them on another machine.
- Some files use older names for the same physical parameter, such as `nu`, `lambda`, or `lambda2`.
- Some scripts use fixed true-value reference lines in plots. Verify the true `lambda_2` value against the dataset before reusing the plots.
- Some scripts define `N_b = 100` but sample a hard-coded number of boundary points internally. Check this before treating `N_b` as the actual boundary sample count.
- The notebooks may contain outputs from previous runs, warning messages, and old experimental notes. They are preserved for transparency rather than polished reproducibility.
- Extensionless files should be renamed with `.py` if they are intended to be executable Python scripts.

## Recommended archive structure

```text
archive/
├── README.md
├── legacy_notebooks/
├── legacy_scripts/
├── plotting_scripts/
├── figures/
└── parameter_stability/
```

Suggested placement:

```text
legacy_notebooks/
    Burgers_Equation.ipynb
    Burgers_Equation-Self_Adaptive.ipynb
    Labelled_Burgers_Equation.ipynb
    Inverse_Labelled_Burgers_Equation.ipynb
    myPINN_inverse_problem*.ipynb
    *_vs_lambda2.ipynb

legacy_scripts/
    labelled_vs_lambda.py
    start_vs_lambda.py
    depth_vs_lambda.py
    width_vs_lambda.py

plotting_scripts/
    general_results_plotting.py
    N_l_plotting.py
    start_lambda_plotting.py
    depth_plotting.py
    width_plotting.py
    plot_excel_start.py

figures/
    Burgers.png
    collocation_points_Burgers.png
    loss.png
    error history.png
    w0.png

parameter_stability/
    params_mc_deviation.py
    variance of parameters.ipynb
    parameters_deviation.png
    parameter_parallel_coordinates.png
```
