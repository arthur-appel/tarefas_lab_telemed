import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from dataset import X_train, y_train, X_teste, y_teste, media_log_sp, std_log_sp
from model import RedeCasas


n_features= X_train.shape[1]

def treinar_modelo(modelo, X_train, y_train, criterio, otimizador, epocas=300):
    modelo.train()
    historico_perda = []

    for epoca in range(epocas):
        previsoes = modelo(X_train)

        loss = criterio(previsoes, y_train)

        otimizador.zero_grad()

        loss.backward()

        otimizador.step()

        historico_perda.append(loss.item())

        if (epoca+1) % 50 == 0:
            print(f"Época [{epoca+1}/{epocas}] - Erro (MSE): {loss.item():.4f}\n")

    return modelo, historico_perda

#loss function
MSE = torch.nn.MSELoss()

torch.manual_seed(42)
modelo1 = RedeCasas(n_entradas=n_features)
adam1 = torch.optim.Adam(modelo1.parameters(), lr=0.001)

print("--- TREINANDO MODELO 1 (lr = 0.001) ---")
modelo1, perdas1 = treinar_modelo(
    modelo=modelo1, 
    X_train=X_train, 
    y_train=y_train, 
    criterio=MSE, 
    otimizador = adam1, 
    epocas=300 )


torch.manual_seed(42)
modelo2 = RedeCasas(n_entradas=n_features)
adam2 = torch.optim.Adam(modelo2.parameters(), lr=0.002)

print("\n--- TREINANDO MODELO 2 (lr = 0.002) ---")
modelo2, perdas2 = treinar_modelo(
    modelo=modelo2,
    X_train=X_train,
    y_train=y_train,
    criterio=MSE,
    otimizador=adam2,
    epocas = 300
)

def reverter_valores(modelo):
    modelo.eval()

    with torch.no_grad():
        
        previsoes_norm = modelo(X_teste)
        
        previsoes_norm = previsoes_norm.numpy()

    previsoes_log = (previsoes_norm * std_log_sp) + media_log_sp

    previsoes_dolares = np.exp(previsoes_log)

    y_real_log = (y_teste.numpy() * std_log_sp) + media_log_sp
    y_real_dolares = np.exp(y_real_log)

    return y_real_dolares, previsoes_dolares

def obter_metricas(modelo):

    y_real, y_previsto = reverter_valores(modelo)

    rmse = np.sqrt(mean_squared_error(y_real, y_previsto))
    mae = mean_absolute_error(y_real, y_previsto)
    r2 = r2_score(y_real, y_previsto)

    print("--- RESULTADOS NO CONJUNTO DE TESTE ---")
    print(f"MAE:  ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R²:    {r2:.4f}")   

    return rmse, mae, r2

print("\n=== AVALIAÇÃO: MODELO 1 (lr=0.001) ===")
obter_metricas(modelo1)

print("=== AVALIAÇÃO: MODELO 2 (lr=0.002) ===")
obter_metricas(modelo2)

#conclusões: learning rate de 0.001 obteve resultados melhores, acima de 0.001 os resultados começam a piorar.
#quanto maior o número de épocas, mais o modelo se aprimora e obtém resultados mais precisos.
