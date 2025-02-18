import os
import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .data_processing import load_data, preprocess_data
from .model import create_lstm_model


def train_and_evaluate(
    symbol="AAPL",
    start_date="2000-01-01",
    lookback=30,
    epochs=10,
    batch_size=32,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
):
    # 1. Load and preprocess data (returns)
    df = load_data(symbol, start_date)
    X, y, scaler = preprocess_data(df, lookback=lookback)

    # 2. Split data
    total_samples = len(X)
    train_idx = int(train_split * total_samples)
    val_idx = train_idx + int(val_split * total_samples)

    X_train, X_val, X_test = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
    y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]

    print(
        f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}"
    )

    # 3. Create and train the model
    model = create_lstm_model(input_shape=(lookback, 1))
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # 4. Predictions and metrics on Test Set
    y_pred_test = model.predict(X_test)

    # Inverse-transform the scaled returns
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_test_inv = scaler.inverse_transform(y_pred_test).flatten()

    # Compute metrics (in return space)
    test_mse = mean_squared_error(y_test_inv, y_pred_test_inv)
    test_rmse = math.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test_inv, y_pred_test_inv)

    print(
        f"Test Results - MSE: {test_mse:.6f}, RMSE: {test_rmse:.6f}, MAE: {test_mae:.6f}"
    )

    # 5. Plot actual vs. predicted returns
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label="Actual Returns", color="blue")
    plt.plot(y_pred_test_inv, label="Predicted Returns", color="red", linestyle="--")
    plt.title("Test Set: Actual vs Predicted Returns")
    plt.xlabel("Sample Index")
    plt.ylabel("Daily Return")
    plt.legend()
    plt.show()

    return {"mse": test_mse, "rmse": test_rmse, "mae": test_mae}


# Example usage
if __name__ == "__main__":
    train_and_evaluate("AAPL", start_date="2015-01-01")
