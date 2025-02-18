import os
from fastapi import FastAPI, Query
from pydantic import BaseModel

# Imports das suas funções
from src.train_and_evaluate import train_and_evaluate
from src.train import train_model
from src.inference import predict_return
from src.data_processing import load_data

app = FastAPI()


class SymbolRequest(BaseModel):
    symbol: str


@app.post("/predict")
def predict_endpoint(req: SymbolRequest):
    """
    Endpoint único que:
    1) Recebe um símbolo de ação.
    2) Verifica se existe modelo salvo em models/<symbol>.
    3) Se existir, apenas faz a inferência.
    4) Se não existir, treina e depois faz a inferência.
    5) Retorna ao usuário o valor de fechamento atual multiplicado pelo retorno previsto.
    """

    symbol = req.symbol.upper()
    model_dir = f"models/{symbol}"
    model_path = f"{model_dir}/model.h5"

    if os.path.exists(model_path):
        # Só inferência
        predicted_return = predict_return(symbol)
    else:
        # Treina e depois infere
        train_model(symbol)  # ou train_and_evaluate(symbol), se preferir
        predicted_return = predict_return(symbol)

    # Agora, pega o último preço de fechamento real para multiplicar
    df = load_data(symbol)
    current_close = df["Close"].iloc[-1]  # Último valor de fechamento

    predicted_close_price = current_close * (1 + predicted_return)

    return {
        "symbol": symbol,
        "current_close": current_close,
        "predicted_return": predicted_return,
        "predicted_close_price": predicted_close_price,
    }
