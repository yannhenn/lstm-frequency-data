import torch
import torch.nn as nn
import os


class LSTMModel(nn.Module):
    """LSTM model for time series prediction."""
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x_lstm, _ = self.lstm(x)
        x_linear = self.linear(x_lstm)
        return x_linear
    
    def save(self, path):
        """Save model state dict.
        
        Args:
            path: File path (without .pth extension)
        """
        torch.save(self.state_dict(), f"{path}.pth")
        print(f"Model saved to {path}.pth")
    
    def load(self, path):
        """Load model state dict.
        
        Args:
            path: File path (without .pth extension)
        """
        checkpoint = torch.load(f"{path}.pth", weights_only=False)
        self.load_state_dict(checkpoint)
        print(f"Model loaded from {path}.pth")
    
    def exists(self, path):
        """Check if a saved model exists at path.
        
        Args:
            path: File path (without .pth extension)
        
        Returns:
            bool: True if model file exists
        """
        return os.path.exists(f"{path}.pth")
