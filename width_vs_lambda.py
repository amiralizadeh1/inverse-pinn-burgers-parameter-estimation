"""
width_vs_lambda.py

Network-width sensitivity study for inverse PINN parameter recovery.

This script trains an inverse Physics-Informed Neural Network (PINN) for the
one-dimensional Burgers equation:

    u_t + u u_x - lambda_2 u_xx = 0

The aim is to study how neural-network width affects recovery of the unknown
physical parameter lambda_2. Several fully connected architectures with the
same depth but different numbers of neurons per hidden layer are defined, and
one selected width case is trained in this script.

For the selected architecture, the script performs 3-fold cross-validation and
records training loss, test loss, relative L2 error, and the recovered lambda_2
trajectory. Results are saved as plots and an Excel file for later comparison
across different network-width cases.

Author: Amir Alizadeh
"""

import tensorflow as tf
import datetime, os, sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import scipy.optimize
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import time
from pyDOE import lhs
import seaborn as sns 
import codecs, json
from sklearn.model_selection import KFold
import pandas as pd

# Hyperparameters.
foldername = 'Revision'
seed = 4
np.random.seed(seed)
tf.random.set_seed(seed)
maxiter = 10000
N_b = 100
N_f = 10000
N_l = 10000
initial_lambda2 = 0.1

network_architectures = [
    np.array([2, 5, 5, 5, 5, 1]),
    np.array([2, 10, 10, 10, 10, 1]),
    np.array([2, 15, 15, 15, 15, 1]),
    np.array([2, 25, 25, 25, 25, 1])
]

# Global plotting style
import matplotlib as mpl
try:
    get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
except Exception:
    pass

mpl.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 15,
    'axes.labelsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'figure.titlesize': 15,
    'figure.figsize': (6.0, 4.0),
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'lines.linewidth': 2.0,
    'lines.markersize': 5.5,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'legend.frameon': False,
    'legend.handlelength': 2.0,
    'image.cmap': 'viridis',
    'figure.autolayout': True,
})

def style_ax(ax, xlabel=None, ylabel=None, title=None, legend=True):
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title:  ax.set_title(title)
    if legend and ax.get_legend() is not None:
        ax.legend()

print("✅")

# Data Prep
baseFolder = os.path.join(os.getcwd(), foldername)
newFolder = os.path.join(baseFolder, 'width_vs_performance')
if not os.path.exists(newFolder): os.makedirs(newFolder)

data = scipy.io.loadmat('Data/burgers_shock_IC_sin2pi.mat')
x = data['x']
t = data['t']
usol = data['usol']

X, T = np.meshgrid(x, t)

print('shapes of data arrays:')
print(X.shape, T.shape, usol.shape)

# Test Data
X_u_test_general = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
lb = X_u_test_general[0]
ub = X_u_test_general[-1]
u_test_general = usol.flatten('F')[:,None]

# Training Data
def All_data(N_f):
    X_all_labelled = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_all_labelled = usol.flatten('F')[:,None]

    # idx_l = np.random.permutation(X_all_labelled.shape[0])
    idx_l = np.random.choice(X_all_labelled.shape[0], N_l, replace=False)

    X_labelled_shuffled = X_all_labelled[idx_l, :]
    u_labelled_shuffled = u_all_labelled[idx_l, :]

    leftedge_x = np.hstack((X[0,:][:,None], T[0,:][:,None]))
    leftedge_u = usol[:,0][:,None]

    bottomedge_x = np.hstack((X[:,0][:,None], T[:,0][:,None]))
    bottomedge_u = usol[-1,:][:,None]

    topedge_x = np.hstack((X[:,-1][:,None], T[:,0][:,None]))
    topedge_u = usol[0,:][:,None]

    X_u_boundary = np.vstack([leftedge_x, bottomedge_x, topedge_x])
    u_boundary = np.vstack([leftedge_u, bottomedge_u, topedge_u])

    idx_b = np.random.choice(X_u_boundary.shape[0], 300, replace=False)

    X_u_boundary_shuffled = X_u_boundary[idx_b, :] 
    u_boundary_shuffled = u_boundary[idx_b,:] 

    X_f_all = lb + (ub-lb)*lhs(2, N_f)
    X_f_all = np.vstack((X_f_all, X_u_boundary_shuffled))

    return X_f_all, X_u_boundary_shuffled, u_boundary_shuffled, X_labelled_shuffled, u_labelled_shuffled

# Inverse Problem PINN
class Sequentialmodel(tf.Module):
    def __init__(self, layers, name=None):
        self.W = []
        self.parameters = 0
        self.iteration = 0
        self.losss_history = np.array([])
        self.error_history = np.array([])
        self.w_history = np.array([])
        self.lambda2_history = np.array([])
        self.loss_test_history = np.array([])
        self.lambda2 = tf.Variable([initial_lambda2], dtype='float64', trainable = True)
        self.parameters += 1
        self.prev_lambda2 = None
        self.stop_due_to_lambda2 = False
        self.loss_val = None
        self.loss_test_val = None
        
        for i in range(len(layers) - 1):
            input_dim = layers[i]
            output_dim = layers[i+1]
            
            std_dv = np.sqrt((2.0/(input_dim + output_dim)))
            w = tf.random.normal([input_dim, output_dim], dtype = 'float64') * std_dv
            w = tf.Variable(w, trainable=True, name = 'w' + str(i+1))
            b = tf.Variable(tf.cast(tf.zeros([output_dim]), dtype = 'float64'), trainable = True, name = 'b' + str(i+1))
            self.W.append(w)
            self.W.append(b)
            self.parameters +=  input_dim * output_dim + output_dim
    
    def forward(self, x):
        x = (x-lb)/(ub-lb)
        a = x
        for i in range(len(layers)-2):
            z = tf.add(tf.matmul(a, self.W[2*i]), self.W[2*i+1])
            a = tf.nn.tanh(z)
        a = tf.add(tf.matmul(a, self.W[-2]), self.W[-1])
        return a
    
    def get_weights(self):
        parameters_1d = []
        
        for i in range (len(layers)-1):
            w_1d = tf.reshape(self.W[2*i],[-1])
            b_1d = tf.reshape(self.W[2*i+1],[-1])
            
            parameters_1d = tf.concat([parameters_1d, w_1d], 0)
            parameters_1d = tf.concat([parameters_1d, b_1d], 0)
        parameters_1d = tf.concat([parameters_1d, self.lambda2], 0)
        return parameters_1d
        
    def set_weights(self, parameters):
        for i in range (len(layers)-1):
            shape_w = tf.shape(self.W[2*i]).numpy()
            size_w = tf.size(self.W[2*i]).numpy()
            
            shape_b = tf.shape(self.W[2*i+1]).numpy()
            size_b = tf.size(self.W[2*i+1]).numpy()
            
            pick_w = parameters[0:size_w]
            self.W[2*i].assign(tf.reshape(pick_w,shape_w))
            parameters = np.delete(parameters,np.arange(size_w),0)
            
            pick_b = parameters[0:size_b]
            self.W[2*i+1].assign(tf.reshape(pick_b, shape_b))
            parameters = np.delete(parameters,np.arange(size_b),0)
            
        self.lambda2.assign(parameters[0:1])
        parameters = np.delete(parameters, np.arange(1), 0)
    
    def loss_labelled(self, x, y):
        loss_l = tf.reduce_mean(tf.square(y-self.forward(x)))
        return loss_l
            
    def loss_BC(self, x, y):
        loss_u = tf.reduce_mean(tf.square(y-self.forward(x)))
        return loss_u

    def loss_PDE(self, x_to_train_f):
        g = tf.Variable(x_to_train_f, dtype = 'float64', trainable = False)
        x_f = g[:, 0:1]
        t_f = g[:, 1:2]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_f)
            tape.watch(t_f)

            g = tf.stack([x_f[:,0], t_f[:,0]], axis=1)
            z = self.forward(g)
            u_x = tape.gradient(z, x_f)

        u_t = tape.gradient(z, t_f)
        u_xx = tape.gradient(u_x, x_f)

        del tape

        f = u_t + (self.forward(g))*(u_x) - self.lambda2*u_xx

        loss_f = tf.reduce_mean(tf.square(f))

        return loss_f
    
    def loss_test(self, x_labelled, y_labelled, x_boundary, y_boundary):
        loss_l = self.loss_labelled(x_labelled, y_labelled)
        loss_u = self.loss_BC(x_boundary, y_boundary)
        loss_test = loss_l + loss_u
        return loss_test
    
    def loss(self, x, y, g, a, b):
        loss_l = self.loss_labelled(a, b)
        loss_u = self.loss_BC(x, y)
        loss_f = self.loss_PDE(g)

        loss = loss_u + loss_f + loss_l
        return loss, loss_u, loss_f, loss_l
    
    def optimizerfunc(self, parameters):
        self.set_weights(parameters)
        self.loss_val = None
       
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            loss_val, loss_u, loss_f, loss_l = self.loss(X_u_train, u_train, X_f_all, X_labelled_train, u_labelled_train)
            self.loss_val = loss_val.numpy()
        grads = tape.gradient(loss_val, self.trainable_variables)
        del tape
        
        grads_1d = []
        
        for i in range (len(layers)-1):
            grads_w_1d = tf.reshape(grads[2*i], [-1])
            grads_b_1d = tf.reshape(grads[2*i+1], [-1])

            grads_1d = tf.concat([grads_1d, grads_w_1d], 0)
            grads_1d = tf.concat([grads_1d, grads_b_1d], 0)
        grads_lambda2_1d = tf.reshape(grads[-1],[-1])
        grads_1d = tf.concat([grads_1d, grads_lambda2_1d], 0)

        return self.loss_val, grads_1d.numpy()
    
    def optimizer_callback(self, parameters):
        self.iteration += 1
        
        u_pred = self.forward(X_labelled_test)
        error = np.linalg.norm((u_labelled_test-u_pred),2)/np.linalg.norm(u_labelled_test,2)
        self.error_history = np.append(self.error_history, error)
         
        self.w_history = np.append(self.w_history, self.W[0][0][0])
        
        self.losss_history = np.append(self.losss_history, self.loss_val)
        self.loss_test_val = self.loss_test(X_labelled_test, u_labelled_test, X_u_test, u_test).numpy()
        self.loss_test_history = np.append(self.loss_test_history, self.loss_test_val)
        print('iteration:', self.iteration, 'lambda2:', self.lambda2[0].numpy(), 'error:', error, 'loss_train:', self.loss_val, 'loss_test:', self.loss_test_val)
        self.lambda2_history = np.append(self.lambda2_history, self.lambda2.numpy())

# Main execution
if __name__ == "__main__":
    start_time = time.time()
    X_f_all, X_u_all, u_all, X_labelled_all, u_labelled_all = All_data(N_f)
    kf = KFold(n_splits=3, shuffle=False, random_state=None)
    
    width_idx = 0
    layers = network_architectures[width_idx]
    width_start = time.time()
    width_folder = os.path.join(newFolder, f'width_{width_idx}_neurons_{layers[1]}')
    if not os.path.exists(width_folder): os.makedirs(width_folder)
    
    fold = 0
    loss_histories = []
    loss_test_histories = []
    lambda_histories = []
    error_histories = []
    
    for (train_idx_u, test_idx_u), (train_idx_l, test_idx_l) in zip(kf.split(X_u_all), kf.split(X_labelled_all)):
        fold += 1
        
        X_u_train = X_u_all[train_idx_u]
        u_train = u_all[train_idx_u]
        X_u_test = X_u_all[test_idx_u]
        u_test = u_all[test_idx_u]
        
        X_labelled_train = X_labelled_all[train_idx_l]
        u_labelled_train = u_labelled_all[train_idx_l]
        X_labelled_test = X_labelled_all[test_idx_l]
        u_labelled_test = u_labelled_all[test_idx_l]
        
        PINN = Sequentialmodel(layers)
        initial_params = PINN.get_weights().numpy()
        
        results = scipy.optimize.minimize(fun = PINN.optimizerfunc,
                                            x0 = initial_params,
                                            args = (),
                                            method='L-BFGS-B',
                                            jac = True,
                                            callback = PINN.optimizer_callback,
                                            options = {'maxiter': maxiter,
                                                    'ftol': 1 * np.finfo(float).eps,
                                                    'gtol': 5e-8,
                                                    'maxfun': 50000,
                                                    'maxcor': 200,
                                                    'maxls': 50})

        loss_histories.append(PINN.losss_history)
        loss_test_histories.append(PINN.loss_test_history)
        lambda_histories.append(PINN.lambda2_history)
        error_histories.append(PINN.error_history)
        print(f'✅ Fold {fold} finished (width={layers[1]})')
    
    loss_array = np.array(loss_histories, dtype=object)
    mean_loss = np.mean([np.array(h) for h in loss_histories], axis=0)
    std_loss = np.std([np.array(h) for h in loss_histories], axis=0)
    iterations = np.arange(len(mean_loss))
    
    loss_test_array = np.array(loss_test_histories, dtype=object)
    mean_loss_test = np.mean([np.array(h) for h in loss_test_histories], axis=0)
    std_loss_test = np.std([np.array(h) for h in loss_test_histories], axis=0)
    
    lambda_array = np.array(lambda_histories, dtype=object)
    mean_lambda = np.mean([np.array(h) for h in lambda_histories], axis=0)
    std_lambda = np.std([np.array(h) for h in lambda_histories], axis=0)
    
    error_array = np.array(error_histories, dtype=object)
    mean_error = np.mean([np.array(h) for h in error_histories], axis=0)
    std_error = np.std([np.array(h) for h in error_histories], axis=0)
    
    # Save to Excel with all folds and means
    data_dict = {'iterations': iterations}
    for fold_idx in range(len(error_histories)):
        data_dict[f'error_fold_{fold_idx}'] = error_histories[fold_idx]
        data_dict[f'lambda_fold_{fold_idx}'] = lambda_histories[fold_idx]
        data_dict[f'loss_test_fold_{fold_idx}'] = loss_test_histories[fold_idx]
        data_dict[f'loss_train_fold_{fold_idx}'] = loss_histories[fold_idx]
    data_dict['error_all_folds'] = mean_error
    data_dict['lambda_all_folds'] = mean_lambda
    data_dict['loss_test_all_folds'] = mean_loss_test
    data_dict['loss_train_all_folds'] = mean_loss
    
    df = pd.DataFrame(data_dict)
    df.to_excel(os.path.join(width_folder, 'results.xlsx'), index=False)
    
    # Plot mean and std of loss curves (training and test)
    fig, ax = plt.subplots()
    ax.semilogy(iterations, mean_loss, label='Mean Training Loss', linewidth=2, color='tab:blue')
    ax.fill_between(iterations, mean_loss - std_loss, mean_loss + std_loss, alpha=0.3, color='tab:blue', label='±1 Std Dev')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig(os.path.join(width_folder, 'loss_train_all_folds.png'))
    plt.close()
    
    fig, ax = plt.subplots()
    ax.semilogy(iterations, mean_loss_test, label='Mean Test Loss', linewidth=2, color='tab:orange')
    ax.fill_between(iterations, mean_loss_test - std_loss_test, mean_loss_test + std_loss_test, alpha=0.3, color='tab:orange', label='±1 Std Dev')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig(os.path.join(width_folder, 'loss_test_all_folds.png'))
    plt.close()
    
    fig, ax = plt.subplots()
    ax.plot(iterations, mean_lambda, label='Mean of Free Parameter ($\\lambda$)', linewidth=2)
    ax.fill_between(iterations, mean_lambda - std_lambda, mean_lambda + std_lambda, alpha=0.3, label='±1 Std Dev')
    ax.axhline(y=0.001/np.pi, color='red', linestyle='--', linewidth=2, label='True Value ($0.005/\\pi$)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Free Parameter ($\\lambda$)')
    ax.legend()
    plt.savefig(os.path.join(width_folder, 'lambda_all_folds.png'))
    plt.close()
    
    fig, ax = plt.subplots()
    ax.semilogy(iterations, mean_error, label='Mean Error', linewidth=2)
    ax.fill_between(iterations, mean_error - std_error, mean_error + std_error, alpha=0.3, label='±1 Std Dev')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative L2 Error')
    ax.legend()
    plt.savefig(os.path.join(width_folder, 'error_all_folds.png'))
    plt.close()
    
    width_elapsed = time.time() - width_start
    print(f'✅ Width {layers[1]} finished ({width_elapsed:.2f}s)')
    print(f'Results saved to {width_folder}')
    sys.exit(0)
