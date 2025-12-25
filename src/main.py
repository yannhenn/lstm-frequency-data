import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib
import pandas as pd
import numpy as np
import os,torch
import torch.optim as optim
import torch.utils.data as data

# Try to use an interactive backend, fall back to Agg if not available
try:
    matplotlib.use('Qt5Agg')  # Try Qt backend first
except ImportError:
    try:
        matplotlib.use('TkAgg')  # Try Tk backend
    except ImportError:
        matplotlib.use('Agg')  # Use non-interactive backend as fallback
        print("Using non-interactive backend - figures will be saved to file")

DATA_DIR = "data/2nd_test/2nd_test/"
CASHED_CALCULATES_FN="calculated_raw_data"
MODEL_SAVE_PATH="model_trained"


def crawl_datafiles(data_dir):
    datafile_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if not file.endswith('.zip'):
                datafile_paths.append(os.path.join(root, file))
    return datafile_paths

def persist_dataframe(filename: str, dataframe: pd.DataFrame):
    df_pickle = dataframe.to_pickle(f"{filename}.pkl")

def calculate_rms(dataset:pd.DataFrame)->float:
    return np.sqrt(np.mean(np.square(dataset)))

def calculate_kurtosis(dataset:pd.DataFrame)->float:
    n = len(dataset)
    mean = np.mean(dataset)
    std = np.std(dataset)
    kurtosis = (np.sum((dataset - mean) ** 4) / n) / (std ** 4)
    return kurtosis


def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    """
    split a multivariate sequence past, future samples (X and y)
    
    """
    X, y = [], []
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)

def create_training_predictions(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature['RMS'].values.astype(np.float32))
        y.append(target['RMS'].values.astype(np.float32))
    npX, npy=np.array(X),np.array(y)
    return torch.tensor(npX), torch.tensor(npy)


## Bearing 1 was defect in Dataset 2
columns=['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']
if(os.path.exists(f"{CASHED_CALCULATES_FN}.pkl")):
    feature_df = pd.read_pickle(f"{CASHED_CALCULATES_FN}.pkl")
else:
    datafile_paths = crawl_datafiles(DATA_DIR)  
    rms_list = list()
    for filename in datafile_paths:
        print(f"Processing file: {filename}", end='\r', flush=True)
        data_in = pd.read_csv(filename, sep='\t', header=None, names=columns)
        timestamp_raw = filename.removeprefix(DATA_DIR).removesuffix(".txt").replace("_",":")
        parts = timestamp_raw.split('.')
        timestamp = pd.to_datetime(f"{parts[0]}-{parts[1]}-{parts[2]} {parts[3]}:{parts[4]}:{parts[5]}")
        data_selected = data_raw = data_in.iloc[:min(2048, len(data_in))]['Bearing 1']
        dataset_features = {
            'Timestamp': timestamp,
            'RMS': calculate_rms(data_selected),
            'Kurtosis': calculate_kurtosis(data_selected)
            }
        rms_list.append(dataset_features)
    feature_df = pd.DataFrame(rms_list)
    persist_dataframe(CASHED_CALCULATES_FN, feature_df)

print(feature_df.head())
print(feature_df.info())

plt.figure(figsize=(15, 5))
feature_df['RMS'].plot(color='blue')
feature_df['Kurtosis'].plot(color='orange')
plt.title(DATA_DIR)
plt.xlabel('Timestamp')
plt.ylabel('Value')

plt.xlabel("Timestamp")
plt.ylabel("Parameter")
# Try to show the figure interactively
if matplotlib.get_backend() != 'Agg':
    plt.show()

train_size = int(len(feature_df) * 0.7)
test_size = len(feature_df) - train_size
train_data = feature_df[:train_size]
test_data = feature_df[train_size:]

lookback = 5
X_train, y_train = create_training_predictions(train_data, lookback=lookback)
X_test, y_test = create_training_predictions(test_data, lookback=lookback)

# Reshape data to add feature dimension: (samples, timesteps, features)
X_train = X_train.unsqueeze(-1)  # Shape: (samples, lookback, 1)
y_train = y_train.unsqueeze(-1)  # Shape: (samples, lookback, 1)
X_test = X_test.unsqueeze(-1)    # Shape: (samples, lookback, 1)
y_test = y_test.unsqueeze(-1)    # Shape: (samples, lookback, 1)

# Create torch dataset
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)  # Changed from lookback to 1
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x_lstm, _ = self.lstm(x)
        x_linear = self.linear(x_lstm)
        return x_linear
    

model = Model()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
 
n_epochs = 10000
if(os.path.exists(f"{MODEL_SAVE_PATH}.pth")):
    checkpoint = torch.load(f"{MODEL_SAVE_PATH}.pth", weights_only=False)
    model.load_state_dict(checkpoint)
else:
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
        # Validation
        if epoch % 100 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
    torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}.pth")

model.eval()

# Plot actual test data
plt.figure(figsize=(15, 8))
test_actual = test_data['RMS'].values
plt.plot(test_actual, c='b', label='Actual RMS', linewidth=2, alpha=0.7)

# Get all predictions at once (more efficient)
with torch.no_grad():
    all_test_preds = model(X_test)  # Shape: (num_test_samples, lookback, 1)
    
    # Extract the last prediction from each sequence
    final_preds = all_test_preds[:, -1, 0].numpy()  # Shape: (num_test_samples,)
    
    # Plot as a continuous line
    x_positions = range(lookback, len(final_preds) + lookback)
    plt.plot(x_positions, final_preds, c='g', alpha=0.7, label='Test Predictions', linewidth=2)

plt.title('RMS Predictions vs Actual - Test Data')
plt.xlabel('Time Steps (relative to test start)')
plt.ylabel('RMS Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# Predict from a specific test sample
def predict_from_sequence(model, input_sequence, n_steps=5):
    """
    Predict n future steps from a given input sequence
    
    Args:
        model: Trained PyTorch model
        input_sequence: torch.Tensor of shape (1, lookback, 1)
        n_steps: Number of future steps to predict
    
    Returns:
        List of predicted values
    """
    model.eval()
    predictions = []
    current_seq = input_sequence.clone()
    
    with torch.no_grad():
        for _ in range(n_steps):
            pred = model(current_seq)[:, -1:, :]
            predictions.append(pred.item())
            current_seq = torch.cat([current_seq[:, 1:, :], pred], dim=1)
    
    return predictions



def rolling_predictions(model, test_data_df, lookback, n_steps_ahead=5):
    """
    Create rolling window predictions for test data
    """
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for i in range(len(test_data_df) - lookback):
            # Get sequence
            sequence = test_data_df.iloc[i:i+lookback]['RMS'].values.astype(np.float32)
            sequence_tensor = torch.tensor(sequence).unsqueeze(0).unsqueeze(-1)
            
            # Predict next n steps
            predictions = []
            current_seq = sequence_tensor.clone()
            
            for _ in range(n_steps_ahead):
                pred = model(current_seq)[:, -1:, :]
                predictions.append(pred.item())
                current_seq = torch.cat([current_seq[:, 1:, :], pred], dim=1)
            
            all_predictions.append(predictions)
    
    return np.array(all_predictions)
# Example: Predict from the 10th test sample
sample_idx = 10
if sample_idx < len(X_test):
    input_seq = X_test[sample_idx:sample_idx+1]
    future_preds = predict_from_sequence(model, input_seq, n_steps=20)
    
    print(f"\nPredictions starting from test sample {sample_idx}:")
    print(future_preds)
# Generate rolling predictions
rolling_preds = rolling_predictions(model, test_data, lookback, n_steps_ahead=5)
print(f"\nGenerated rolling predictions shape: {rolling_preds.shape}")
print(f"First 3 predictions (5 steps each):\n{rolling_preds[:3]}")

# Visualize rolling predictions
n_steps_ahead = 5
plt.figure(figsize=(20, 10))

# Plot 1: Overview of all rolling predictions
plt.subplot(2, 1, 1)
test_actual = test_data['RMS'].values
plt.plot(test_actual, c='b', label='Actual Test Data', linewidth=2, alpha=0.7)

# Plot each rolling prediction
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

# Plot 2: Detailed view of specific predictions
plt.subplot(2, 1, 2)

# Show 3 example predictions in detail
examples = [0, len(rolling_preds)//2, min(len(rolling_preds)-1, len(rolling_preds)-10)]
colors = ['red', 'purple', 'orange']

for idx, (example_idx, color) in enumerate(zip(examples, colors)):
    # Get the input sequence
    input_start = example_idx
    input_end = example_idx + lookback
    input_seq = test_actual[input_start:input_end]
    
    # Get the prediction
    pred_start = input_end
    pred_end = pred_start + n_steps_ahead
    prediction = rolling_preds[example_idx]
    
    # Get the actual future values
    actual_future = test_actual[pred_start:pred_end]
    
    # Plot input sequence
    plt.plot(range(input_start, input_end), input_seq, 
             c=color, linewidth=2, marker='o', markersize=4,
             label=f'Input Seq {example_idx}')
    
    # Plot prediction
    plt.plot(range(pred_start, pred_end), prediction, 
             c=color, linewidth=2, linestyle='--', marker='s', markersize=4,
             label=f'Prediction {example_idx}')
    
    # Plot actual future values
    plt.plot(range(pred_start, pred_end), actual_future, 
             c='black', linewidth=1, linestyle=':', marker='x', markersize=6,
             alpha=0.5)

# Add actual data as background
plt.plot(test_actual, c='lightblue', linewidth=1, alpha=0.3, label='Actual (background)')

plt.title('Detailed Rolling Predictions - Input Sequence → Predictions vs Actual')
plt.xlabel('Time Steps')
plt.ylabel('RMS Value')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot 3: Prediction accuracy over time
plt.figure(figsize=(15, 6))

# Calculate errors for each prediction step
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

# Plot error distribution for each prediction step
plt.subplot(1, 2, 1)
for step in range(n_steps_ahead):
    plt.plot(errors_by_step[step], alpha=0.5, label=f'{step+1}-step ahead')

plt.title('Absolute Prediction Errors Over Time')
plt.xlabel('Rolling Window Index')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot average error by prediction horizon
plt.subplot(1, 2, 2)
mean_errors = [np.mean(errors) for errors in errors_by_step]
std_errors = [np.std(errors) for errors in errors_by_step]

plt.bar(range(1, n_steps_ahead+1), mean_errors, yerr=std_errors, 
        alpha=0.7, capsize=5, color='coral')
plt.title('Average Error by Prediction Horizon')
plt.xlabel('Steps Ahead')
plt.ylabel('Mean Absolute Error')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Print statistics
print("\n=== Prediction Statistics ===")
for step in range(n_steps_ahead):
    mean_err = np.mean(errors_by_step[step])
    std_err = np.std(errors_by_step[step])
    print(f"{step+1}-step ahead: MAE = {mean_err:.6f}, Std = {std_err:.6f}")

