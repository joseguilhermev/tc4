import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from .data_processing import load_data, preprocess_data


def predict_return(symbol: str, lookback: int = 30):
    """
    Carrega o modelo salvo e faz predição do próximo retorno
    com base na última sequência de dados.
    """
    model_path = f"models/{symbol}/model.keras"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}")

    # Carrega modelo
    model = load_model(model_path)

    # Carrega dados e reprocessa
    df = load_data(symbol)
    X, _, _ = preprocess_data(df, lookback)

    # Pega a última sequência
    last_sequence = X[-1]  # shape (lookback, 1)
    last_sequence = np.expand_dims(last_sequence, axis=0)  # (1, lookback, 1)

    predicted_return = model.predict(last_sequence)
    # predicted_return é um array 2D, ex: [[0.015]] -> pegamos o valor
    predicted_return_value = predicted_return[0, 0]

    # Se quiser reverter o scaling (caso tenha sido salvo):
    # from joblib import load
    # scaler = load(f"models/{symbol}/scaler.joblib")
    # predicted_return_value = scaler.inverse_transform([[predicted_return_value]])[0][0]

    return predicted_return_value


print(predict_return("AAPL"))  # Exemplo de uso
