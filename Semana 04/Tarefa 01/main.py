import numpy as np
import pandas as pd
from functools import partial
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch

from dataset import X_train, y_train, X_teste, y_teste, media_log_sp, std_log_sp
from model import RedeCasas, RedeCasasRes, RedeCasasResLN, RedeCasasDrop, RedeCasasResLNDrop

n_features = X_train.shape[1]
criterio = torch.nn.MSELoss()
SEEDS = [0, 1, 2, 3, 4]

experimentos = [
    {"nome": "Modelo 1 (Base)",          "classe": RedeCasas,                       "lr": 0.0010, "epocas": 400},
    {"nome": "Modelo 2 (Base)",          "classe": RedeCasas,                       "lr": 0.0020, "epocas": 400},
    {"nome": "Modelo 3 (ResNet)",        "classe": RedeCasasRes,                    "lr": 0.0010, "epocas": 500},
    {"nome": "Modelo 4 (ResNet)",        "classe": RedeCasasRes,                    "lr": 0.0020, "epocas": 800},
    {"nome": "Modelo 5 (Res+LN)",        "classe": RedeCasasResLN,                  "lr": 0.0010, "epocas": 500},
    {"nome": "Modelo 6 (Drop 0.1)",      "classe": RedeCasasDrop,                   "lr": 0.0010, "epocas": 500},
    {"nome": "Modelo 7 (Res+LN+Drop)",   "classe": RedeCasasResLNDrop,              "lr": 0.0010, "epocas": 500},

]


def treinar_modelo(modelo, X_train, y_train, criterio, otimizador, epocas):
    modelo.train()
    for epoca in range(epocas):
        previsoes = modelo(X_train)
        loss = criterio(previsoes, y_train)

        otimizador.zero_grad()
        loss.backward()
        otimizador.step()
    return modelo


def avaliar_conjunto(modelo, X_tensor, y_tensor):
    modelo.eval()
    with torch.no_grad():
        p_norm = modelo(X_tensor).numpy()

    p_dolares = np.exp((p_norm * std_log_sp) + media_log_sp)
    y_dolares = np.exp((y_tensor.numpy() * std_log_sp) + media_log_sp)

    mae = mean_absolute_error(y_dolares, p_dolares)
    rmse = np.sqrt(mean_squared_error(y_dolares, p_dolares))
    r2 = r2_score(y_dolares, p_dolares)
    return mae, rmse, r2


def mean_std(valores, fmt):
    return f"{format(np.mean(valores), fmt)} ± {format(np.std(valores), fmt)}"


tabela_resultados = []
melhor_run = {"r2": -np.inf, "modelo": None, "seed": None}      # melhor execução individual
melhor_media = {"r2": -np.inf, "modelo": None}                  # melhor média entre seeds

for exp in experimentos:
    print(f">>> TREINANDO: {exp['nome']} | LR={exp['lr']} | Épocas={exp['epocas']} | {len(SEEDS)} seeds")

    r2_trs, r2_tes, maes_te, rmses_te = [], [], [], []

    for seed in SEEDS:
        torch.manual_seed(seed)
        modelo = exp["classe"](n_entradas=n_features)
        otimizador = torch.optim.Adam(modelo.parameters(), lr=exp["lr"])

        treinar_modelo(modelo, X_train, y_train, criterio, otimizador, exp["epocas"])

        _, _, r2_tr = avaliar_conjunto(modelo, X_train, y_train)
        mae_te, rmse_te, r2_te = avaliar_conjunto(modelo, X_teste, y_teste)

        r2_trs.append(r2_tr)
        r2_tes.append(r2_te)
        maes_te.append(mae_te)
        rmses_te.append(rmse_te)

        if r2_te > melhor_run["r2"]:
            melhor_run = {"r2": r2_te, "modelo": exp["nome"], "seed": seed}

    media_r2 = np.mean(r2_tes)
    if media_r2 > melhor_media["r2"]:
        melhor_media = {"r2": media_r2, "modelo": exp["nome"]}

    tabela_resultados.append({
        "Modelo": exp["nome"],
        "LR": exp["lr"],
        "Épocas": exp["epocas"],
        "R² Treino": mean_std(r2_trs, ".4f"),
        "R² Teste": mean_std(r2_tes, ".4f"),
        "R² Teste (máx)": f"{np.max(r2_tes):.4f}",
        "MAE Teste ($)": mean_std(maes_te, ",.0f"),
        "RMSE Teste ($)": mean_std(rmses_te, ",.0f"),
    })

df_resultados = pd.DataFrame(tabela_resultados)
print("\n" + "=" * 130)
print(f"   QUADRO COMPARATIVO (média ± desvio de {len(SEEDS)} seeds)")
print("=" * 130)
print(df_resultados.to_string(index=False))
print("=" * 130)

print("\n" + "#" * 70)
print(f"  >>> MAIOR R² DE TESTE (execução única): {melhor_run['r2']:.4f}")
print(f"      Modelo: {melhor_run['modelo']} | Seed: {melhor_run['seed']}")
print(f"  >>> MAIOR R² DE TESTE (média das seeds): {melhor_media['r2']:.4f}")
print(f"      Modelo: {melhor_media['modelo']}")
print("#" * 70)