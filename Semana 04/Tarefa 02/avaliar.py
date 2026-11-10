import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import v2
from sklearn.metrics import classification_report

from data_treatment import modelo, media_treino, std_treino, device

# 1. carregando os dados do teste
dados_test = np.load('Semana 04/Tarefa 02/data/Dataset5_test_1ch.npz')

test_x = torch.from_numpy(dados_test['image']).permute(0, 3, 1, 2)
test_y = torch.from_numpy(dados_test['image_label'].flatten()).long()

test_dataset = TensorDataset(test_x, test_y)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. transformando o teste (sem flip dessa vez)
transform_test = v2.Compose([
    v2.ToDtype(torch.float32, scale=False),
    v2.Normalize(mean=[media_treino], std=[std_treino]),
])

# 3. carregando os pesos da resnet já treinada (.pth)
caminho_pesos = "Semana 04/Tarefa 02/resnet18_raiox.pth"
modelo.load_state_dict(torch.load(caminho_pesos, map_location=device))
modelo.eval()  

todos_previstos = []
todos_reais = []

# 4. passando as imagens de teste na rede
print("Avaliando o modelo no conjunto de Teste...")
with torch.no_grad():
    for imagens, labels in test_loader:
        imagens = imagens.to(device)
        imagens = transform_test(imagens)
        
        saidas = modelo(imagens)
        _, previstos = torch.max(saidas, dim=1)
        
        todos_previstos.extend(previstos.cpu().numpy())
        todos_reais.extend(labels.numpy())

# 5. precisão, recall e f1 score por classe
nomes_classes = [
    "0: COVID-19",
    "1: Lung-Opacity",
    "2: Normal",
    "3: Viral Pneumonia",
    "4: Tuberculosis"
]

print("\n=== RELATÓRIO DE CLASSIFICAÇÃO (CONJUNTO DE TESTE) ===")
print(classification_report(todos_reais, todos_previstos, target_names=nomes_classes, digits=4))