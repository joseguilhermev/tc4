import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(symbol: str, start_date: str = "2000-01-01"):
    """
    Carrega o histórico completo da ação desde a data de início fornecida
    (idealmente, o IPO). Retorna um DataFrame do pandas.
    """
    df = yf.download(symbol, start=start_date, auto_adjust=True)
    # Opcional: filtrar colunas ou remover NaNs
    df.dropna(inplace=True)
    return df


def preprocess_data(df: pd.DataFrame, lookback: int = 30):
    """
    Transforma colunas de preço em retorno e realiza a padronização.
    Returns = (Close - Open) / Open
    Cria janelas de tamanho `lookback` (30 dias por padrão) para LSTM.

    Retorna (X, y, scaler) prontos para uso em treinamento.
    """
    # Calcula retornos
    df["Return"] = (df["Close"] - df["Open"]) / df["Open"]

    # Padronização
    scaler = StandardScaler()
    df["Return_scaled"] = scaler.fit_transform(df[["Return"]])

    # Criando janelas de 30 dias para o LSTM
    X, y = [], []
    data = df["Return_scaled"].values

    for i in range(lookback, len(data)):
        X.append(data[i - lookback : i])  # Pegamos os últimos 30 dias
        y.append(data[i])  # Prever o retorno do próximo dia

    # Convertendo para numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Reshape para o formato esperado pelo LSTM: [samples, timesteps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler
