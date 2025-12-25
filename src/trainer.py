import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


class Trainer:
    """Handles model training."""
    
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
    
    def train(self, X_train, y_train, X_test, y_test, n_epochs=10000, batch_size=8, 
              save_path=None, print_every=100):
        """Train the model.
        
        Args:
            X_train, y_train: Training data tensors
            X_test, y_test: Test data tensors for validation
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            save_path: Path to save model (without extension)
            print_every: Print progress every n epochs
        """
        loader = data.DataLoader(
            data.TensorDataset(X_train, y_train),
            shuffle=True,
            batch_size=batch_size
        )
        
        for epoch in range(n_epochs):
            self.model.train()
            for X_batch, y_batch in loader:
                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            if epoch % print_every != 0:
                continue
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(X_train)
                train_rmse = np.sqrt(self.loss_fn(y_pred, y_train).item())
                y_pred = self.model(X_test)
                test_rmse = np.sqrt(self.loss_fn(y_pred, y_test).item())
            print(f"Epoch {epoch}: train RMSE {train_rmse:.4f}, test RMSE {test_rmse:.4f}")
        
        if save_path:
            self.save_model(save_path)
    
    def save_model(self, path):
        """Save model state dict."""
        torch.save(self.model.state_dict(), f"{path}.pth")
        print(f"Model saved to {path}.pth")
    
    def load_model(self, path):
        """Load model state dict."""
        checkpoint = torch.load(f"{path}.pth", weights_only=False)
        self.model.load_state_dict(checkpoint)
        print(f"Model loaded from {path}.pth")
    
    def load_or_train(self, X_train, y_train, X_test, y_test, save_path, **train_kwargs):
        """Load existing model or train a new one."""
        if os.path.exists(f"{save_path}.pth"):
            self.load_model(save_path)
        else:
            self.train(X_train, y_train, X_test, y_test, save_path=save_path, **train_kwargs)
