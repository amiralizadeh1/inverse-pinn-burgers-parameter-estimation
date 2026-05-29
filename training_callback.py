import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt

class TrainingCallback:
    def __init__(self, model, X_u_train, u_train, X_f, X_labelled, u_labelled, X_test, u_test, print_every=50):
        self.model = model
        self.X_u_train = X_u_train
        self.u_train = u_train
        self.X_f = X_f
        self.X_labelled = X_labelled
        self.u_labelled = u_labelled
        self.X_test = X_test
        self.u_test = u_test
        self.print_every = print_every
        
        self.iteration = 0
        self.loss_history = []
        self.loss_u_history = []
        self.loss_f_history = []
        self.loss_l_history = []
        self.lambda2_history = []
        self.error_history = []
    
    def __call__(self, parameters):
        self.iteration += 1
        
        # Compute losses
        loss_total, loss_u, loss_f, loss_l = self.model.loss(
            self.X_u_train, self.u_train, self.X_f, self.X_labelled, self.u_labelled
        )
        
        # Compute test error
        u_pred = self.model.forward(self.X_test)
        error = np.linalg.norm(self.u_test - u_pred, 2) / np.linalg.norm(self.u_test, 2)
        
        # Store history
        self.loss_history.append(loss_total.numpy())
        self.loss_u_history.append(loss_u.numpy())
        self.loss_f_history.append(loss_f.numpy())
        self.loss_l_history.append(loss_l.numpy())
        self.lambda2_history.append(self.model.lambda2.numpy()[0])
        self.error_history.append(error)
        
        # Print progress
        if self.iteration % self.print_every == 0 or self.iteration == 1:
            print(f"Iter {self.iteration:5d} | Loss: {loss_total:.6e} | "
                  f"Loss_u: {loss_u:.6e} | Loss_f: {loss_f:.6e} | Loss_l: {loss_l:.6e} | "
                  f"λ2: {self.model.lambda2.numpy()[0]:.6f} | Error: {error:.6f}")
    
    def plot_history(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Total loss
        axes[0, 0].semilogy(self.loss_history, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True)
        
        # Individual losses
        axes[0, 1].semilogy(self.loss_u_history, label='Loss_u (BC)', linewidth=2)
        axes[0, 1].semilogy(self.loss_f_history, label='Loss_f (PDE)', linewidth=2)
        axes[0, 1].semilogy(self.loss_l_history, label='Loss_l (Data)', linewidth=2)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss Components')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Lambda2 evolution
        axes[1, 0].plot(self.lambda2_history, 'r-', linewidth=2)
        axes[1, 0].axhline(y=0.00318, color='k', linestyle='--', label='True λ2')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('λ2')
        axes[1, 0].set_title('λ2 Evolution')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Test error
        axes[1, 1].semilogy(self.error_history, 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Relative L2 Error')
        axes[1, 1].set_title('Test Error')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
