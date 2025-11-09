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


def crawl_datafiles(data_dir):
    datafile_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if not file.endswith('.zip'):
                datafile_paths.append(os.path.join(root, file))
    return datafile_paths

def calculate_rms(dataset:pd.DataFrame)->float:
    return np.sqrt(np.mean(np.square(dataset)))

def calculate_kurtosis(dataset:pd.DataFrame)->float:
    n = len(dataset)
    mean = np.mean(dataset)
    std = np.std(dataset)
    kurtosis = (np.sum((dataset - mean) ** 4) / n) / (std ** 4)
    return kurtosis

def create_dataset(dataset, lookback):
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
    return torch.tensor(X), torch.tensor(y)

datafile_paths = crawl_datafiles(DATA_DIR)  
## Bearing 1 was defect in Dataset 2
columns=['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']

rms_list = list()
for filename in datafile_paths:
    print(f"Processing file: {filename}", end='\r', flush=True)
    data_in = pd.read_csv(filename, sep='\t', header=None, names=columns)
    timestamp_raw = filename.removeprefix(DATA_DIR).removesuffix(".txt").replace("_",":")
    parts = timestamp_raw.split('.')
    timestamp = pd.to_datetime(f"{parts[0]}-{parts[1]}-{parts[2]} {parts[3]}:{parts[4]}:{parts[5]}")
    data_selected = data_raw = data_in.iloc[:min(2048, len(data_in))]['Bearing 1']
    dataset_keyfigures = {
        'Timestamp': timestamp,
        'RMS': calculate_rms(data_selected),
        'Kurtosis': calculate_kurtosis(data_selected)
        }
    rms_list.append(dataset_keyfigures)
rms_df = pd.DataFrame(rms_list)
print(rms_df)

plt.figure(figsize=(15, 5))
rms_df['RMS'].plot(color='blue')
rms_df['Kurtosis'].plot(color='orange')
plt.title(DATA_DIR)
plt.xlabel('Timestamp')
plt.ylabel('Value')

plt.xlabel("Timestamp")
plt.ylabel("Parameter")
# Try to show the figure interactively
if matplotlib.get_backend() != 'Agg':
    plt.show()
train_size = int(len(rms_df) * 0.7)
test = len(rms_df) - train_size
train_data = rms_df[:train_size]
test_data = rms_df[train_size:]

lookback = 1
X_train, y_train = create_dataset(train_data, lookback=lookback)
X_test, y_test = create_dataset(test_data, lookback=lookback)
# Create torch dataset
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
    

model = Model()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
 
n_epochs = 2000
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
 
with torch.no_grad():
    # shift train predictions for plotting
    train_plot = np.ones_like(rms_df) * np.nan
    y_pred = model(X_train)
    y_pred = y_pred[:, -1, :]
    train_plot[lookback:train_size] = model(X_train)[:, -1, :]
    # shift test predictions for plotting
    test_plot = np.ones_like(rms_df) * np.nan
    test_plot[train_size+lookback:len(rms_df)] = model(X_test)[:, -1, :]
# plot
plt.plot(rms_df['RMS'], c='b')
plt.plot(train_plot, c='r')
plt.plot(test_plot, c='g')
plt.show()