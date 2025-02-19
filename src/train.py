# src/train.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)

from src.data_processing import load_and_preprocess_data
from src.model import build_lstm_model

import tensorflow as tf


def train_model(
    csv_path="data/AAPL/AAPL.csv",
    model_save_path="models/lstm_model.keras",
    feature_columns=None,
    target_column="Close",
    window_size=20,
    test_size=0.2,
    lstm_units=64,
    dropout_rate=0.2,
    loss="mean_squared_error",
    optimizer="adam",
    epochs=10,
    batch_size=32,
    verbose=1,
):
    # 1. Carrega e pré-processa os dados
    X, y, scaler = load_and_preprocess_data(
        csv_path, feature_columns, target_column, window_size
    )

    # 2. Divide em treino/validação
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    # 3. Constrói o modelo
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        loss=loss,
        optimizer=optimizer,
    )

    # 4. Treina o modelo
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    # 5. Avalia o modelo
    predictions_val = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions_val)
    rmse = root_mean_squared_error(y_val, predictions_val)
    mape = mean_absolute_percentage_error(y_val, predictions_val)
    print(f"Validation MAPE: {mape:.4f}")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")

    # 6. Salva o modelo
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Modelo salvo em: {model_save_path}")

    return model, scaler, history


if __name__ == "__main__":
    train_model(
        csv_path="data/AAPL/AAPL.csv",
        model_save_path="models/lstm_model.keras",
        epochs=100,
        batch_size=32,
        verbose=1,
    )
