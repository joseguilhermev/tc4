# src/model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_lstm_model(
    input_shape,
    lstm_units=64,
    dropout_rate=0.2,
    loss="mean_squared_error",
    optimizer="adam",
):
    """
    Constroi e compila o modelo LSTM para previsão de séries temporais.

    Parâmetros:
    -----------
    input_shape : tuple
        Formato de entrada (window_size, num_features).
    lstm_units : int
        Número de unidades LSTM.
    dropout_rate : float
        Taxa de dropout aplicada após a camada LSTM.
    loss : str
        Função de perda, por exemplo 'mean_squared_error' ou 'mean_absolute_error'.
    optimizer : str
        Otimizador, por exemplo 'adam'.

    Retorna:
    --------
    model : keras.Model
        Modelo LSTM compilado.
    """
    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # saída para prever um único valor (ex.: preço)

    model.compile(loss=loss, optimizer=optimizer)
    return model
