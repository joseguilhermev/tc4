import requests
import pandas as pd


def main():
    # 1. Ler o CSV
    df = pd.read_csv("data/AAPL/AAPL.csv")

    # 2. Selecionar as colunas e pegar as últimas 20 linhas
    feature_cols = ["Open", "High", "Low", "Close", "Volume"]
    recent_data_df = df[feature_cols].tail(20)

    # 3. Converter em lista de listas (shape: 20 x 5)
    recent_data_list = recent_data_df.values.tolist()

    # 4. Montar o payload
    payload = {"recent_data": recent_data_list}

    # 5. Fazer a requisição POST
    url = "http://localhost:8000/predict"  # Ajuste se necessário
    response = requests.post(url, json=payload)

    # 6. Exibir resultado
    if response.status_code == 200:
        print("Resposta da API:", response.json())
    else:
        print(f"Erro {response.status_code}: {response.text}")


if __name__ == "__main__":
    main()
