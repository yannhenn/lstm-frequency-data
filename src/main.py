import matplotlib.pyplot as plt
import matplotlib

# Try to use an interactive backend
try:
    matplotlib.use('Qt5Agg')
except ImportError:
    try:
        matplotlib.use('TkAgg')
    except ImportError:
        matplotlib.use('Agg')
        print("Using non-interactive backend - figures will be saved to file")

from model import LSTMModel
from data_utils import load_or_create_features, prepare_data
from trainer import Trainer
from inference import Predictor, Evaluator

# Configuration
DATA_DIR = "data/2nd_test/2nd_test/"
CACHED_CALCULATES_FN = "calculated_raw_data"
MODEL_SAVE_PATH = "model_trained"
COLUMNS = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']
LOOKBACK = 5
N_EPOCHS = 10000
N_STEPS_AHEAD = 5


def main():
    # Load or create features
    feature_df = load_or_create_features(DATA_DIR, CACHED_CALCULATES_FN, COLUMNS)
    print(feature_df.head())
    print(feature_df.info())
    
    # Plot raw data
    plt.figure(figsize=(15, 5))
    feature_df['RMS'].plot(color='blue')
    feature_df['Kurtosis'].plot(color='orange')
    plt.title(DATA_DIR)
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    if matplotlib.get_backend() != 'Agg':
        plt.show()
    
    # Prepare data
    data = prepare_data(feature_df, LOOKBACK)
    
    # Initialize model
    model = LSTMModel()
    
    # Load or train model
    if model.exists(MODEL_SAVE_PATH):
        model.load(MODEL_SAVE_PATH)
    else:
        trainer = Trainer(model)
        trainer.train(
            data['X_train'], data['y_train'],
            data['X_test'], data['y_test'],
            n_epochs=N_EPOCHS
        )
        model.save(MODEL_SAVE_PATH)
    
    # Initialize predictor and evaluator
    predictor = Predictor(model)
    evaluator = Evaluator(predictor)
    
    # Plot test predictions
    evaluator.plot_test_predictions(data['test_data'], data['X_test'], LOOKBACK)
    
    # Example: Predict from specific test sample
    sample_idx = 10
    if sample_idx < len(data['X_test']):
        input_seq = data['X_test'][sample_idx:sample_idx + 1]
        future_preds = predictor.predict_from_sequence(input_seq, n_steps=20)
        print(f"\nPredictions starting from test sample {sample_idx}:")
        print(future_preds)
    
    # Generate and visualize rolling predictions
    rolling_preds = predictor.rolling_predictions(data['test_data'], LOOKBACK, N_STEPS_AHEAD)
    print(f"\nGenerated rolling predictions shape: {rolling_preds.shape}")
    print(f"First 3 predictions ({N_STEPS_AHEAD} steps each):\n{rolling_preds[:3]}")
    
    evaluator.plot_rolling_predictions(data['test_data'], rolling_preds, LOOKBACK, N_STEPS_AHEAD)
    
    # Analyze prediction errors
    errors_by_step = evaluator.plot_prediction_errors(
        data['test_data'], rolling_preds, LOOKBACK, N_STEPS_AHEAD
    )
    evaluator.print_statistics(errors_by_step, N_STEPS_AHEAD)


if __name__ == "__main__":
    main()

