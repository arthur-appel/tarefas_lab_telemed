import zipfile
import struct
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import v2
from torchvision.models import resnet18, resnet34, resnet50
import torch.nn as nn

# 1. Carregando os dados crus.

#dados_raw_train = np.load('Semana 04/Tarefa 02/data/raw/Dataset5_raw_train.npz')

#dados_raw_test = np.load('Semana 04/Tarefa 02/data/raw/Dataset5_raw_test.npz')

#dados_raw_val = np.load('Semana 04/Tarefa 02/data/raw/Dataset5_raw_val.npz')


# 2. Remoção de canais, normalização entre 0 e 1. 

def converter_dataset_1canal(arquivo_entrada, arquivo_saida, chunk_size=1000):
    print(f"\n--- Processando: {arquivo_entrada} -> {arquivo_saida} ---")
    
    # 2.1 carregando labels e nomes diretamente 
    raw_data = np.load(arquivo_entrada)
    labels = raw_data['image_label']
    names = raw_data['image_name']
    total_imgs = len(labels)
    
    # 2.2 pré alocando memória 
    imagens_1ch = np.empty((total_imgs, 224, 224, 1), dtype=np.float32)
    
    # 2.3 lendo as images em lotes
    bytes_por_img = 224 * 224 * 3 * 8  # 3 canais float64
    

    with zipfile.ZipFile(arquivo_entrada, 'r') as z:
        with z.open('image.npy') as f:
            # pula o cabeçalho do arquivo npy
            v = f.read(8)[6:8]
            h_len = struct.unpack('<H', f.read(2))[0] if v[0] == 1 else struct.unpack('<I', f.read(4))[0]
            f.read(h_len)
            
            # processa em blocos de 'chunk_size' imagens
            for i in range(0, total_imgs, chunk_size):
                n_imgs = min(chunk_size, total_imgs - i)
                raw_bytes = f.read(n_imgs * bytes_por_img)
                
                # deserializa o lote
                lote = np.frombuffer(raw_bytes, dtype=np.float64).reshape((n_imgs, 224, 224, 3))
                
                # aqui é a parte mais importante, pegando apenas o Canal 0 e normaliza entre 0 e 1 em float32
                lote_1ch = (lote[:, :, :, 0:1] / 255.0).astype(np.float32)
                
                imagens_1ch[i : i + n_imgs] = lote_1ch
                print(f"Progresso: {i + n_imgs}/{total_imgs} imagens processadas...", end='\r')
                
    print(f"\nSalvando {arquivo_saida}...")
    np.savez_compressed(
        arquivo_saida,
        image=imagens_1ch,
        image_label=labels,
        image_name=names
    )
    print(f"Sucesso! {arquivo_saida} salvo.")



# 3. Carregando os dados convertidos e calculando métricas.

dados_train = np.load('Semana 04/Tarefa 02/data/Dataset5_train_1ch.npz')

train_x = torch.from_numpy(dados_train['image']).permute(0,3,1,2)
train_y = torch.from_numpy(dados_train['image_label'].flatten()).long()

media_treino = float(train_x.mean())
std_treino = float(train_x.std())

#dados_test = np.load('Semana 04/Tarefa 02/data/Dataset5_test_1ch.npz')

#dados_val = np.load('Semana 04/Tarefa 02/data/Dataset5_val_1ch.npz')


#print("Treino:", dados_train['image'].shape, dados_train['image'].dtype)
#print("Validação:", dados_val['image'].shape, dados_val['image'].dtype)
#print("Teste:", dados_test['image'].shape, dados_test['image'].dtype)

# VALIDADO

# 3.1 Conferindo se os valores estão normalizados entre 0 e 1:

#print(f"Mínimo: {dados_train['image'][:100].min():.2f}, Máximo: {dados_train['image'].max():.2f}")

# VALIDADO

# 4. Função para normalizar o dataset com z-score, float32 e horizontal flip de 50%. 

transform_train = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=False),
    v2.Normalize(mean=[media_treino], std=[std_treino]),
])


# 5. Criando o Dataset e o DataLoader, Modelo e o adaptando.

train_dataset = TensorDataset(train_x, train_y)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
modelo = resnet18(weights=None)

modelo.conv1 = nn.Conv2d(
    in_channels=1,            #1 canal (grayscale)
    out_channels=64,
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False
)

num_features = modelo.fc.in_features
modelo.fc = nn.Linear(num_features, 5)  #adaptando pra sair as nossas 5 features possiveis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = modelo.to(device)

criterio = nn.CrossEntropyLoss()
otimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)


# 6 Aplicando o treinamento da rede.

if __name__ == "__main__":
    num_epocas = 5

    print(f"Iniciando o treinamento:\n\n")

    for epoca in range (num_epocas):
        modelo.train()
        soma_loss = 0.0
        acertos = 0
        total = 0

        for batch_idx, (imagens, labels) in enumerate(train_loader):

            imagens, labels = imagens.to(device), labels.to(device)
            
            # 6.1 aplicando o flip aleatório (0.5)
            imagens = transform_train(imagens)
            
            # 6.2 passos do treino
            otimizador.zero_grad()              # 1. zerando gradientes anteriores
            saidas = modelo(imagens)            # 2. forward pass (chute da rede)
            loss = criterio(saidas, labels)     # 3. calculando o erro com a loss 
            loss.backward()                     # 4. backpropagation (calculando derivadas)
            otimizador.step()                   # 5. atualizando pesos

            # 6.3 estatisticas

            soma_loss += loss.item()
            _, previstos = torch.max(saidas, dim=1) # pega a classe com maior nota (0 a 4)
            acertos += (previstos == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(f"Época [{epoca+1}/{num_epocas}] | Lote [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        loss_media = soma_loss / len(train_loader)
        acuracia = (acertos / total) * 100
        print(f"==> Fim da Época {epoca+1} | Loss Média: {loss_media:.4f} | Acurácia Treino: {acuracia:.2f}%\n")

    torch.save(modelo.state_dict(), "Semana 04/Tarefa 02/resnet18_raiox.pth")

  
    




