import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch

from dataset import X_train, y_train, X_teste, y_teste, media_log_sp, std_log_sp
from model import RedeCasas, RedeCasasRes, RedeCasasResLN, RedeCasasDrop, RedeCasasResLNDrop

n_features = X_train.shape[1]
criterio = torch.nn.MSELoss()

# Configs
experimentos = [
    {"nome": "Modelo 1 (Base)",   "classe": RedeCasas,    "lr": 0.0010, "epocas": 400},
    {"nome": "Modelo 2 (Base)",   "classe": RedeCasas,    "lr": 0.0020, "epocas": 600},
    {"nome": "Modelo 3 (ResNet)", "classe": RedeCasasRes, "lr": 0.0010, "epocas": 400},
    {"nome": "Modelo 4 (ResNet)", "classe": RedeCasasRes, "lr": 0.0025, "epocas": 800},
    {"nome": "Modelo 5 (Res+LN)",  "classe": RedeCasasResLN, "lr": 0.0025, "epocas": 400},
    {"nome": "Modelo 6 (Drop)", "classe": RedeCasasDrop, "lr": 0.0010, "epocas": 600},
    {"nome": "Modelo 7 (Res+LN+Drop)", "classe": RedeCasasResLNDrop, "lr": 0.0010, "epocas": 400},
]

# Functions

def treinar_modelo(modelo, X_train, y_train, criterio, otimizador, epocas):
    modelo.train()
    historico_perda = []

    for epoca in range(epocas):
        previsoes = modelo(X_train)
        loss = criterio(previsoes, y_train)

        otimizador.zero_grad()
        loss.backward()
        otimizador.step()

        historico_perda.append(loss.item())

        if (epoca + 1) % 100 == 0 or (epoca + 1) == epocas:
            print(f"  Época [{epoca+1:3d}/{epocas}] - Erro (MSE): {loss.item():.4f}")

    return modelo, historico_perda


def avaliar_modelo(modelo):
    modelo.eval()
    with torch.no_grad():
        previsoes_norm = modelo(X_teste).numpy()

    # Reverte a escala (log-zscore para dólares)
    previsoes_log = (previsoes_norm * std_log_sp) + media_log_sp
    previsoes_dolares = np.exp(previsoes_log)

    y_real_log = (y_teste.numpy() * std_log_sp) + media_log_sp
    y_real_dolares = np.exp(y_real_log)

    mae = mean_absolute_error(y_real_dolares, previsoes_dolares)
    rmse = np.sqrt(mean_squared_error(y_real_dolares, previsoes_dolares))
    r2 = r2_score(y_real_dolares, previsoes_dolares)

    return mae, rmse, r2


# Execução
tabela_resultados = []

for exp in experimentos:
    print(f"\n>>> TREINANDO: {exp['nome']} | LR={exp['lr']} | Épocas={exp['epocas']}")
    
    # Seed individual por modelo para reprodutibilidade estrita
    torch.manual_seed(42)
    modelo = exp["classe"](n_entradas=n_features)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=exp["lr"])

    treinar_modelo(modelo, X_train, y_train, criterio, otimizador, exp["epocas"])
    mae, rmse, r2 = avaliar_modelo(modelo)

    tabela_resultados.append({
        "Modelo": exp["nome"],
        "Arquitetura": exp["classe"].__name__,
        "LR": exp["lr"],
        "Épocas": exp["epocas"],
        "MAE ($)": f"{mae:,.2f}",
        "RMSE ($)": f"{rmse:,.2f}",
        "R²": f"{r2:.4f}"
    })

# Exibição consolidada no terminal
df_resultados = pd.DataFrame(tabela_resultados)
print("\n" + "=" * 76)
print("                   QUADRO COMPARATIVO DOS MODELOS                   ")
print("=" * 76)
print(df_resultados.to_string(index=False))
print("=" * 76)