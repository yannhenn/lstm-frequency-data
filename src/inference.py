import numpy as np
import torch
import matplotlib.pyplot as plt


class Predictor:
    """Handles model inference and predictions."""
    
    def __init__(self, model):
        self.model = model
    
    def predict_from_sequence(self, input_sequence, n_steps=5):
        """Predict n future steps from a given input sequence.
        
        Args:
            input_sequence: torch.Tensor of shape (1, lookback, 1)
            n_steps: Number of future steps to predict
        
        Returns:
            List of predicted values
        """
        self.model.eval()
        predictions = []
        current_seq = input_sequence.clone()
        
        with torch.no_grad():
            for _ in range(n_steps):
                pred = self.model(current_seq)[:, -1:, :]
                predictions.append(pred.item())
                current_seq = torch.cat([current_seq[:, 1:, :], pred], dim=1)
        
        return predictions
    
    def rolling_predictions(self, test_data_df, lookback, n_steps_ahead=5):
        """Create rolling window predictions for test data."""
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for i in range(len(test_data_df) - lookback):
                sequence = test_data_df.iloc[i:i + lookback]['RMS'].values.astype(np.float32)
                sequence_tensor = torch.tensor(sequence).unsqueeze(0).unsqueeze(-1)
                
                predictions = []
                current_seq = sequence_tensor.clone()
                
                for _ in range(n_steps_ahead):
                    pred = self.model(current_seq)[:, -1:, :]
                    predictions.append(pred.item())
                    current_seq = torch.cat([current_seq[:, 1:, :], pred], dim=1)
                
                all_predictions.append(predictions)
        
        return np.array(all_predictions)
    
    def get_test_predictions(self, X_test):
        """Get predictions for all test samples."""
        self.model.eval()
        with torch.no_grad():
            return self.model(X_test)


class Evaluator:
    """Handles model evaluation and visualization."""
    
    def __init__(self, predictor):
        self.predictor = predictor
    
    def plot_test_predictions(self, test_data, X_test, lookback):
        """Plot actual test data vs predictions."""
        plt.figure(figsize=(15, 8))
        test_actual = test_data['RMS'].values
        plt.plot(test_actual, c='b', label='Actual RMS', linewidth=2, alpha=0.7)
        
        all_test_preds = self.predictor.get_test_predictions(X_test)
        final_preds = all_test_preds[:, -1, 0].numpy()
        
        x_positions = range(lookback, len(final_preds) + lookback)
        plt.plot(x_positions, final_preds, c='g', alpha=0.7, label='Test Predictions', linewidth=2)
        
        plt.title('RMS Predictions vs Actual - Test Data')
        plt.xlabel('Time Steps (relative to test start)')
        plt.ylabel('RMS Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_rolling_predictions(self, test_data, rolling_preds, lookback, n_steps_ahead):
        """Visualize rolling predictions."""
        test_actual = test_data['RMS'].values
        
        plt.figure(figsize=(20, 10))
        
        # Plot 1: Overview
        plt.subplot(2, 1, 1)
        plt.plot(test_actual, c='b', label='Actual Test Data', linewidth=2, alpha=0.7)
        
        for i in range(len(rolling_preds)):
            start_idx = i + lookback
            end_idx = start_idx + n_steps_ahead
            x_positions = range(start_idx, end_idx)
            
            if i == 0:
                plt.plot(x_positions, rolling_preds[i], c='g', alpha=0.2, label='Rolling Predictions')
            else:
                plt.plot(x_positions, rolling_preds[i], c='g', alpha=0.2)
        
        plt.title('Rolling Window Predictions - Complete Test Set')
        plt.xlabel('Time Steps')
        plt.ylabel('RMS Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Detailed view
        plt.subplot(2, 1, 2)
        examples = [0, len(rolling_preds) // 2, min(len(rolling_preds) - 1, len(rolling_preds) - 10)]
        colors = ['red', 'purple', 'orange']
        
        for example_idx, color in zip(examples, colors):
            input_start = example_idx
            input_end = example_idx + lookback
            input_seq = test_actual[input_start:input_end]
            
            pred_start = input_end
            pred_end = pred_start + n_steps_ahead
            prediction = rolling_preds[example_idx]
            actual_future = test_actual[pred_start:pred_end]
            
            plt.plot(range(input_start, input_end), input_seq,
                     c=color, linewidth=2, marker='o', markersize=4,
                     label=f'Input Seq {example_idx}')
            plt.plot(range(pred_start, pred_end), prediction,
                     c=color, linewidth=2, linestyle='--', marker='s', markersize=4,
                     label=f'Prediction {example_idx}')
            plt.plot(range(pred_start, pred_end), actual_future,
                     c='black', linewidth=1, linestyle=':', marker='x', markersize=6,
                     alpha=0.5)
        
        plt.plot(test_actual, c='lightblue', linewidth=1, alpha=0.3, label='Actual (background)')
        plt.title('Detailed Rolling Predictions - Input Sequence → Predictions vs Actual')
        plt.xlabel('Time Steps')
        plt.ylabel('RMS Value')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_errors(self, test_data, rolling_preds, lookback, n_steps_ahead):
        """Plot prediction accuracy analysis."""
        test_actual = test_data['RMS'].values
        
        errors_by_step = []
        for step in range(n_steps_ahead):
            step_errors = []
            for i in range(len(rolling_preds)):
                pred_idx = i + lookback + step
                if pred_idx < len(test_actual):
                    actual_value = test_actual[pred_idx]
                    predicted_value = rolling_preds[i][step]
                    step_errors.append(abs(actual_value - predicted_value))
            errors_by_step.append(step_errors)
        
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        for step in range(n_steps_ahead):
            plt.plot(errors_by_step[step], alpha=0.5, label=f'{step + 1}-step ahead')
        plt.title('Absolute Prediction Errors Over Time')
        plt.xlabel('Rolling Window Index')
        plt.ylabel('Absolute Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        mean_errors = [np.mean(errors) for errors in errors_by_step]
        std_errors = [np.std(errors) for errors in errors_by_step]
        
        plt.bar(range(1, n_steps_ahead + 1), mean_errors, yerr=std_errors,
                alpha=0.7, capsize=5, color='coral')
        plt.title('Average Error by Prediction Horizon')
        plt.xlabel('Steps Ahead')
        plt.ylabel('Mean Absolute Error')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        return errors_by_step
    
    def print_statistics(self, errors_by_step, n_steps_ahead):
        """Print prediction statistics."""
        print("\n=== Prediction Statistics ===")
        for step in range(n_steps_ahead):
            mean_err = np.mean(errors_by_step[step])
            std_err = np.std(errors_by_step[step])
            print(f"{step + 1}-step ahead: MAE = {mean_err:.6f}, Std = {std_err:.6f}")
