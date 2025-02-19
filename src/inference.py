# src/inference.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model


def load_trained_model(model_path="models/lstm_model.keras"):
    model = load_model(model_path)
    return model


def make_inference(
    model,
    scaler,
    recent_data,
    window_size=20,
    feature_columns=None,
    target_column="Close",
):
    if feature_columns is None:
        feature_columns = ["Open", "High", "Low", "Close", "Volume"]

    # Converte a lista de listas em DataFrame
    df_recent = pd.DataFrame(recent_data, columns=feature_columns)

    # Normalização
    scaled_recent = scaler.transform(df_recent[feature_columns].values)

    x_input = scaled_recent[-window_size:]
    x_input = np.expand_dims(x_input, axis=0)

    predicted_scaled = model.predict(x_input)

    target_index = feature_columns.index(target_column)

    # Reconstruindo dummy para inverter normalização somente na posição do alvo
    dummy = np.zeros((1, len(feature_columns)))
    dummy[0, target_index] = predicted_scaled[0, 0]

    inv_scale = scaler.inverse_transform(dummy)
    prediction = inv_scale[0, target_index]

    return float(prediction)
