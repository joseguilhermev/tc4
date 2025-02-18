import os
from .data_processing import load_data, preprocess_data
from .model import create_lstm_model
import tensorflow as tf


def train_model(
    symbol: str,
    start_date: str = "2000-01-01",
    lookback: int = 10,
    epochs: int = 5,
    batch_size: int = 32,
):
    """
    Realiza apenas o treinamento (sem avaliação) e salva o modelo em models/<symbol>.
    """
    df = load_data(symbol, start_date)
    X, y, scaler = preprocess_data(df, lookback)

    # Exemplo: usar todo X e y para treino
    model = create_lstm_model(input_shape=(X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    model_dir = f"models/{symbol}"
    os.makedirs(model_dir, exist_ok=True)
    model.save(f"{model_dir}/model.keras")

    print("Modelo treinado e salvo com sucesso em", model_dir)
