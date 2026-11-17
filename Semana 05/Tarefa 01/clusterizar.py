import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import v2
from sklearn.metrics import classification_report
import torch.nn as nn
from sklearn.cluster import KMeans

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
caminho_pesos = "Semana 04/Tarefa 02/resnet18_raiox_15epocas.pth"
modelo.load_state_dict(torch.load(caminho_pesos, map_location=device))
modelo.fc = nn.Identity()
modelo.eval()  

todos_embeddings = []
todos_reais = []

# 4. passando as imagens de teste na rede
with torch.no_grad():
    for imagens, labels in test_loader:
        imagens = imagens.to(device)
        imagens = transform_test(imagens)
        
        embeddings = modelo(imagens)
        
        todos_embeddings.append(embeddings.cpu().numpy())
        todos_reais.extend(labels.numpy())

todos_embeddings = np.vstack(todos_embeddings)
todos_reais = np.array(todos_reais)

# print(f"Formato da matriz de embeddings extraídos: {todos_embeddings.shape}") - output = (6540, 512), tudo certo.

print("\nRodando o K-Means com K=5...")
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters_kmeans = kmeans.fit_predict(todos_embeddings)

nomes_classes = [
    "0: COVID-19",
    "1: Lung-Opacity",
    "2: Normal",
    "3: Viral Pneumonia",
    "4: Tuberculosis"
]

previstos_kmeans = np.zeros_like(clusters_kmeans)
print("\n=== COMPOSIÇÃO DE CADA CLUSTER DO K-MEANS ===")
for cluster_id in range(5):
    labels_no_cluster = todos_reais[clusters_kmeans == cluster_id]
    
    contagem = np.bincount(labels_no_cluster, minlength=5)
    classe_dominante = np.argmax(contagem)
    
    previstos_kmeans[clusters_kmeans == cluster_id] = classe_dominante
    
    total_cluster = len(labels_no_cluster)
    pureza = (contagem[classe_dominante] / total_cluster) * 100
    print(f"Cluster {cluster_id} ({total_cluster} imgs) -> Dominado por '{nomes_classes[classe_dominante]}' ({pureza:.1f}% das imagens deste grupo)")


print("\n=== RELATÓRIO DE CLASSIFICAÇÃO DO K-MEANS (K=5) ===")
print(classification_report(todos_reais, previstos_kmeans, target_names=nomes_classes, digits=4))


