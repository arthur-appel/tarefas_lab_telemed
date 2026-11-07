import zipfile
import struct
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import v2

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


# 5. Criando o Dataset e o DataLoader

train_dataset = TensorDataset(train_x, train_y)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 6. Aplicando o treinamento da rede.

#for imagens, labels in train_loader:
    #imagens = transform_train(imagens)