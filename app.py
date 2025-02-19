from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

# Importes do seu código
from src.inference import load_trained_model, make_inference
from src.data_processing import load_and_preprocess_data


# Modelo para descrever o request body
class PredictRequest(BaseModel):
    recent_data: list[list[float]]


app = FastAPI(title="LSTM Inference API")

# Carrega modelo e scaler
model = load_trained_model("models/lstm_model.keras")
_, _, scaler = load_and_preprocess_data(
    csv_path="data/AAPL/AAPL.csv",
    feature_columns=["Open", "High", "Low", "Close", "Volume"],
    target_column="Close",
    window_size=20,
)


@app.post("/predict")
def predict_next_close_price(data: PredictRequest):
    """
    Recebe um JSON com o campo "recent_data", que é lista de listas
    (20 linhas, 5 colunas), e retorna a previsão do próximo Close.
    """
    # Acesso via data.recent_data
    prediction = make_inference(
        model=model,
        scaler=scaler,
        recent_data=data.recent_data,
        window_size=20,
        feature_columns=["Open", "High", "Low", "Close", "Volume"],
        target_column="Close",
    )
    return {"Predicted next Close price": prediction}
