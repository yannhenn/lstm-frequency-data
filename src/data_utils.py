import os
import numpy as np
import pandas as pd
import torch


def crawl_datafiles(data_dir):
    """Crawl directory for data files."""
    datafile_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if not file.endswith('.zip'):
                datafile_paths.append(os.path.join(root, file))
    return datafile_paths


def persist_dataframe(filename: str, dataframe: pd.DataFrame):
    """Save dataframe to pickle file."""
    dataframe.to_pickle(f"{filename}.pkl")


def calculate_rms(dataset: pd.DataFrame) -> float:
    """Calculate Root Mean Square."""
    return np.sqrt(np.mean(np.square(dataset)))


def calculate_kurtosis(dataset: pd.DataFrame) -> float:
    """Calculate Kurtosis."""
    n = len(dataset)
    mean = np.mean(dataset)
    std = np.std(dataset)
    kurtosis = (np.sum((dataset - mean) ** 4) / n) / (std ** 4)
    return kurtosis


def create_training_predictions(dataset, lookback):
    """Transform a time series into a prediction dataset.
    
    Args:
        dataset: A DataFrame with 'RMS' column
        lookback: Size of window for prediction
    
    Returns:
        Tuple of (X, y) tensors
    """
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i:i + lookback]
        target = dataset[i + 1:i + lookback + 1]
        X.append(feature['RMS'].values.astype(np.float32))
        y.append(target['RMS'].values.astype(np.float32))
    return torch.tensor(np.array(X)), torch.tensor(np.array(y))


def load_or_create_features(data_dir, cached_filename, columns):
    """Load cached features or create from raw data."""
    if os.path.exists(f"{cached_filename}.pkl"):
        return pd.read_pickle(f"{cached_filename}.pkl")
    
    datafile_paths = crawl_datafiles(data_dir)
    rms_list = []
    
    for filename in datafile_paths:
        print(f"Processing file: {filename}", end='\r', flush=True)
        data_in = pd.read_csv(filename, sep='\t', header=None, names=columns)
        
        timestamp_raw = filename.removeprefix(data_dir).removesuffix(".txt").replace("_", ":")
        parts = timestamp_raw.split('.')
        timestamp = pd.to_datetime(f"{parts[0]}-{parts[1]}-{parts[2]} {parts[3]}:{parts[4]}:{parts[5]}")
        
        data_selected = data_in.iloc[:min(2048, len(data_in))]['Bearing 1']
        dataset_features = {
            'Timestamp': timestamp,
            'RMS': calculate_rms(data_selected),
            'Kurtosis': calculate_kurtosis(data_selected)
        }
        rms_list.append(dataset_features)
    
    feature_df = pd.DataFrame(rms_list)
    persist_dataframe(cached_filename, feature_df)
    return feature_df


def prepare_data(feature_df, lookback, train_ratio=0.7):
    """Prepare training and testing data.
    
    Returns:
        Dictionary with train/test data and tensors
    """
    train_size = int(len(feature_df) * train_ratio)
    train_data = feature_df[:train_size]
    test_data = feature_df[train_size:]
    
    X_train, y_train = create_training_predictions(train_data, lookback=lookback)
    X_test, y_test = create_training_predictions(test_data, lookback=lookback)
    
    # Reshape data to add feature dimension: (samples, timesteps, features)
    X_train = X_train.unsqueeze(-1)
    y_train = y_train.unsqueeze(-1)
    X_test = X_test.unsqueeze(-1)
    y_test = y_test.unsqueeze(-1)
    
    return {
        'train_data': train_data,
        'test_data': test_data,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }
