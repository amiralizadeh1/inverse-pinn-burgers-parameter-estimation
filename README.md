# Inverse PINN Burgers Equation Parameter Estimation

This repository contains research code for an inverse **Physics-Informed Neural Network (PINN)** study on the one-dimensional Burgers equation. The central task is to recover an unknown physical parameter, \(\lambda_2\), by combining sparse labelled data, boundary/initial-condition data, and a PDE-residual loss.

The governing equation used in the scripts is:

\[
u_t + u u_x - \lambda_2 u_{xx} = 0
\]

The main idea follows the PINN framework introduced by Raissi, Perdikaris, and Karniadakis, where neural networks are trained while respecting physical laws expressed as nonlinear partial differential equations. In this repository, that idea is used as an inverse parameter-estimation workflow: the neural network approximates \(u(x,t)\), while \(\lambda_2\) is treated as a trainable variable.

## Research context

This code was developed as part of my PhD work on **physics-informed data-driven modelling**. The Burgers-equation PINN is used as a controlled test case to study how physical constraints, labelled data, network architecture, and optimisation interact in inverse model calibration.

The PINN loss combines:

1. **Boundary and initial-condition loss**  
   Enforces known values at the spatial boundaries and at the initial time.

2. **Labelled-data loss**  
   Matches selected interior solution values from the reference Burgers-equation dataset.

3. **Physics-residual loss**  
   Uses automatic differentiation to calculate \(u_t\), \(u_x\), and \(u_{xx}\), then penalises the Burgers-equation residual.

The repository includes sensitivity studies for:

- number of labelled interior data points,
- neural-network depth,
- neural-network width,
- initial guess of the trainable physical parameter,
- cross-validation stability across folds.

## Main files

### Core inverse PINN implementation

| File | Purpose |
|---|---|
| `Inverse_Labelled_Burgers_Equation.py` | Main cleaned Python script for the inverse PINN Burgers-equation experiment. It loads the Burgers dataset, samples labelled, boundary, and collocation points, trains the inverse PINN with L-BFGS-B, performs 3-fold cross-validation, tracks training loss, test loss, relative L2 error, and the recovered \(\lambda_2\), and exports plots plus `results.xlsx`. |
| `Inverse_Labelled_Burgers_Equation.ipynb` | Notebook version of the inverse labelled PINN workflow. Useful for interactive inspection, development, and reproducing the experiment step by step. |
| `training_callback.py` | Utility callback class for tracking optimisation progress. It stores total loss, boundary loss, PDE-residual loss, labelled-data loss, \(\lambda_2\), and relative L2 error during training, and can plot these histories. |

### Labelled-data sensitivity

| File | Purpose |
|---|---|
| `#labelled_vs_lambda` | Training script for the labelled-data sensitivity study. It runs the inverse PINN for one selected value of \(N_l\), where \(N_l\) is the number of labelled interior data points. It saves fold-wise results under `Revision/performance_vs_N_l/N_l_*`. Rename this file to `labelled_vs_lambda.py` before publishing. |
| `N_l_plotting.py` | Reads the `results.xlsx` files from all `N_l_*` folders and compares how the number of labelled data points affects error, recovered \(\lambda_2\), training loss, and test loss. It creates comparison plots across all labelled-data cases. |
| `#labelled_vs_lambda2.ipynb` | Older notebook version of the labelled-data sensitivity experiment. Kept for development history and interactive review. |

### Network-depth sensitivity

| File | Purpose |
|---|---|
| `depth_vs_lambda.py` | Training script for the network-depth sensitivity study. It defines several architectures with different numbers of hidden layers, selects one depth case, trains it using 3-fold cross-validation, and saves fold-wise results under `Revision/depth_vs_performance/depth_*`. |
| `depth_plotting.py` | Reads all depth-study `results.xlsx` files and generates comparison plots showing how network depth affects error, recovered \(\lambda_2\), training loss, and test loss. |
| `depth_vs_lambda2.ipynb` | Older notebook version of the depth-sensitivity experiment. Useful for checking earlier exploratory runs. |

### Network-width sensitivity

| File | Purpose |
|---|---|
| `width_vs_lambda.py` | Training script for the network-width sensitivity study. It defines architectures with the same depth but different numbers of neurons per hidden layer, selects one width case, trains it with 3-fold cross-validation, and saves results under `Revision/width_vs_performance/width_*`. |
| `width_plotting.py` | Reads all width-study `results.xlsx` files and creates comparison plots across different hidden-layer widths. |
| `width_vs_lambda2.ipynb` | Older notebook version of the width-sensitivity experiment. |
| `width_plotting` | Extensionless/empty or temporary file shown in the working directory. It should be removed, renamed, or moved to `Archive/` before publishing. |

### Initial-lambda sensitivity

| File | Purpose |
|---|---|
| `start_vs_lambda` | Training script for the initialisation sensitivity study. It tests several starting values for \(\lambda_2\), including close underestimates, close overestimates, extreme overestimates, extreme underestimates, and a nonphysical negative initial guess. It performs 3-fold cross-validation for each initial value and saves results under `Revision/start_vs_lambda/lambda_*`. Rename this file to `start_vs_lambda.py` before publishing. |
| `start_lambda_plotting.py` | Reads all initial-lambda `results.xlsx` files and compares error, recovered \(\lambda_2\), test loss, and training loss across initial guesses. |
| `start_vs_lambda2.ipynb` | Older notebook version of the initial-lambda sensitivity experiment. |
| `plot_excel_start.py` | Standalone plotting script for reading an older Excel result file and plotting \(\lambda_2\) histories from different initialisations. It is useful as a legacy post-processing script but is not required for the main workflow. |

### General plotting and result visualisation

| File | Purpose |
|---|---|
| `general_results_plotting.py` | Reads a baseline `results.xlsx` file and creates mean ± standard-deviation plots across folds for relative L2 error, recovered \(\lambda_2\), test loss, and training loss. |
| `collocation_points_Burgers.png` | Figure showing the sampling structure of the PINN training data: residual collocation points \(N_f\), boundary/initial points \(N_b\), and labelled interior points \(N_l\). |
| `Burgers.png` | Visualisation of the Burgers-equation solution field used as the reference dataset. |
| `loss.png` | Saved loss-convergence plot from an earlier or baseline PINN run. |
| `error history.png` | Saved relative-error history from an earlier or baseline PINN run. |
| `w0.png` | Saved diagnostic figure tracking one representative neural-network weight during training. |

### Auxiliary parameter-analysis files

These files are not required for running the inverse Burgers PINN, but they are related to the broader PhD modelling workflow and parameter-stability analysis.

| File | Purpose |
|---|---|
| `params_mc_deviation.py` | Auxiliary script for comparing parameter values across Monte Carlo or repeated calibration runs. It stores regularised and non-regularised parameter arrays, computes mean and standard deviation, and creates comparison figures. |
| `parameter_parallel_coordinates.png` | Figure generated from parameter-comparison analysis, showing how fitted parameters vary between regularisation settings. |
| `parameters_deviation.png` | Figure summarising parameter deviations or uncertainty across repeated runs. |
| `variance of parameters.ipynb` | Notebook for calculating and comparing variance of fitted parameters across optimisation methods or repeated experiments. |

### Legacy notebooks in the main directory

These notebooks are useful for transparency and development history, but they should not be treated as the main supported entry point.

| File | Purpose |
|---|---|
| `Burgers_Equation.ipynb` | Early Burgers-equation PINN notebook, likely focused on the base forward or inverse Burgers setup before the later labelled-data workflow was cleaned. |
| `Burgers_Equation-Self_Adaptive.ipynb` | Experimental self-adaptive PINN notebook, likely exploring adaptive loss weighting or related training modifications. |
| `Labelled_Burgers_Equation.ipynb` | Earlier labelled-data Burgers PINN notebook. It predates the cleaner script-based workflow. |
| `myPINN_inverse_problem.ipynb` | Early inverse PINN notebook. Useful for development history but not recommended as the main reproducible script. |
| `myPINN_inverse_problem_ADAM.ipynb` | Inverse PINN notebook using Adam-based optimisation. Useful for comparing first-order gradient-based training with the later L-BFGS-B workflow. |
| `myPINN_inverse_problem_LBFGS.ipynb` | Inverse PINN notebook using L-BFGS-B optimisation. This is a precursor to the cleaned script-based version. |
| `seed_vs_lambda2.ipynb` | Notebook studying how random seed or initialisation variability affects recovered \(\lambda_2\). |
| `~Inverse_Labelled_Burgers_Equation.ipynb` | Temporary or autosave-style notebook file. This should not be committed to the public repository. Move it to `Archive/` or delete it after verifying that no unique work is inside. |

### Data and generated-output folders

| Folder | Purpose |
|---|---|
| `Data/` | Contains Burgers-equation `.mat` datasets used by the scripts. Check the dataset source/licence before publishing the data publicly. If the data cannot be redistributed, keep only a `Data/README.md` explaining where users should obtain it. |
| `Revision/` | Generated-output folder containing experiment results, plots, Excel summaries, and logs. This is useful locally but should usually be excluded from Git using `.gitignore`, unless selected final figures are intentionally included. |
| `Archive/` | Stores old scripts, old notebooks, partial experiments, and files that are not part of the main reproducible workflow. |
| `.ipynb_checkpoints/` | Jupyter-generated checkpoint folder. This should be ignored by Git. |
| `.git/` | Git metadata folder. Do not edit manually. |

## Suggeste repository structure

For a clean public repository, the current files can be organised as:

```text
inverse-pinn-burgers-parameter-estimation/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/
│   └── Inverse_Labelled_Burgers_Equation.py
│
├── studies/
│   ├── labelled_vs_lambda.py
│   ├── depth_vs_lambda.py
│   ├── width_vs_lambda.py
│   └── start_vs_lambda.py
│
├── plotting/
│   ├── general_results_plotting.py
│   ├── N_l_plotting.py
│   ├── depth_plotting.py
│   ├── width_plotting.py
│   ├── start_lambda_plotting.py
│   └── plot_excel_start.py
│
├── notebooks/
│   └── legacy_notebooks/
│
├── figures/
│   ├── Burgers.png
│   ├── collocation_points_Burgers.png
│   ├── loss.png
│   ├── error history.png
│   └── w0.png
│
├── data/
│   └── README.md
│
└── archive/
    └── README.md
```

## Installation

Create a fresh Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

A typical `requirements.txt` for these scripts should include:

```text
tensorflow
numpy
scipy
matplotlib
pandas
pyDOE
scikit-learn
openpyxl
seaborn
```

## Basic usage

Run the main inverse PINN experiment:

```bash
python Inverse_Labelled_Burgers_Equation.py
```

Run one selected labelled-data sensitivity case:

```bash
python labelled_vs_lambda.py
```

Run one selected depth case:

```bash
python depth_vs_lambda.py
```

Run one selected width case:

```bash
python width_vs_lambda.py
```

Run the initial-lambda sensitivity study:

```bash
python start_vs_lambda.py
```

Generate comparison plots after the corresponding experiments have produced their `results.xlsx` files:

```bash
python N_l_plotting.py
python depth_plotting.py
python width_plotting.py
python start_lambda_plotting.py
```

## Method reference

The main PINN idea is based on:

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).  
**Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.**  
*Journal of Computational Physics*, 378, 686–707.  
https://doi.org/10.1016/j.jcp.2018.10.045

## Project citation

If you use this repository, please cite the associated PhD work or publication when available.

```bibtex
@misc{alizadeh_inverse_pinn_burgers,
  author = {Alizadeh, Amir},
  title = {Inverse PINN Burgers Equation Parameter Estimation},
  year = {2026},
  note = {Research code for inverse Physics-Informed Neural Network parameter recovery}
}
```
