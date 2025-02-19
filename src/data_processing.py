# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess_data(
    csv_path: str, feature_columns=None, target_column="Close", window_size=60
):
    """
    Carrega os dados de um arquivo CSV, realiza limpeza básica,
    normalização e cria janelas temporais (X, y) para treinamento.

    Parâmetros:
    -----------
    csv_path : str
        Caminho para o arquivo CSV (ex: data/AAPL/AAPL.csv).
    feature_columns : list
        Lista com o nome das colunas que serão utilizadas como features.
        Se None, serão utilizadas ['Open', 'High', 'Low', 'Close', 'Volume'] por padrão.
    target_column : str
        Nome da coluna que será predita (por padrão, 'Close').
    window_size : int
        Tamanho da janela temporal para a LSTM (ex.: 60 dias).

    Retorna:
    --------
    X : np.ndarray
        Dados de entrada no formato (amostras, window_size, num_features).
    y : np.ndarray
        Valores de saída correspondentes.
    scaler : MinMaxScaler
        Scaler treinado nos dados (para poder reverter a normalização depois, se necessário).
    """
    if feature_columns is None:
        feature_columns = ["Open", "High", "Low", "Close", "Volume"]

    # 1. Carregar dados
    df = pd.read_csv(csv_path, parse_dates=True, infer_datetime_format=True)

    # 2. Limpeza básica (ex.: remover NA)
    df.dropna(inplace=True)

    # 3. Selecionar colunas de interesse
    df_features = df[feature_columns].copy()

    # 4. Normalizar dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_features)

    # 5. Criar janelas temporais
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size : i])
        # vamos assumir que target_column é parte de feature_columns
        # e queremos prever a próxima posição dessa coluna
        target_index = feature_columns.index(target_column)
        y.append(scaled_data[i, target_index])

    X, y = np.array(X), np.array(y)

    return X, y, scaler
