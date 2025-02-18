import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_data():
    """
    Carrega o histórico completo da ação desde a data de início fornecida
    (idealmente, o IPO). Retorna um DataFrame do pandas com os retornos diários.
    """
    # Se você preferir usar yfinance para baixar os dados, descomente:
    # df = yf.download(symbol, start=start_date, auto_adjust=True)
    # df.dropna(inplace=True)
    # return df

    # Lê os dados do CSV
    df = pd.read_csv(r"C:\Users\joseg\tc4\data\AAPL\AAPL.csv")
    
    # Converte a coluna 'Date' para o tipo datetime e ordena os dados por data
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    
    # Calcula o retorno diário usando o preço ajustado ('Adj Close')
    # O retorno diário é calculado como (Preço de hoje / Preço de ontem) - 1
    df["Return"] = df["Adj Close"].pct_change().fillna(0)  # Fill NaNs to avoid issues

    # Opcional: Calcula o retorno acumulado
    # df["Cumulative Return"] = (1 + df["Return"]).cumprod() - 1

    # Remove a primeira linha, pois o retorno será NaN
    df.dropna(inplace=True)
    
    return df


def preprocess_data(df, lookback=30):
    """
    Processes data: normalizes, creates lookback sequences.
    Returns X, y, and two scalers (one for X, one for y).
    """
    # Assuming 'returns' column in df
    data = df["Return"].values.reshape(-1, 1)

    # Separate scalers for X and y
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # Fit scaler only on training portion (to avoid data leakage)
    train_size = int(0.7 * len(data))
    X_scaler.fit(data[:train_size])  # Fit on training set only
    y_scaler.fit(data[:train_size])  # Fit on training set only

    # Normalize entire dataset using fitted scalers
    data_scaled = X_scaler.transform(data)

    # Create lookback sequences
    X, y = [], []
    for i in range(len(data_scaled) - lookback):
        X.append(data_scaled[i : i + lookback])
        y.append(data_scaled[i + lookback])

    return np.array(X), np.array(y), X_scaler, y_scaler