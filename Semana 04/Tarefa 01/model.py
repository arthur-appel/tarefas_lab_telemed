import torch

class RedeCasas(torch.nn.Module):
    def __init__(self, n_entradas):
        super().__init__()

        #camada 1 - 128 neuronios
        self.linear1 = torch.nn.Linear(in_features=n_entradas, out_features=128)
        self.relu1 = torch.nn.ReLU()


        self.linear2 = torch.nn.Linear(in_features=128, out_features=32)
        self.relu2 = torch.nn.ReLU()

        #saída
        self.saida = torch.nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)

        x = self.linear2(x)
        x = self.relu2(x)

        x = self.saida(x)
        
        return x

class RedeCasasRes(torch.nn.Module):
    def __init__(self, n_entradas):
        super().__init__()

        # Entrada: projeta de n_entradas (182) para o espaço latente de 128
        self.entrada = torch.nn.Linear(n_entradas, 128)
        self.relu = torch.nn.ReLU()
        
        #residual (entra 128 e sai 128)
        self.res_linear1 = torch.nn.Linear(128, 128)
        self.res_linear2 = torch.nn.Linear(128, 128)
        
      
        self.refino = torch.nn.Linear(128, 32)
        self.saida = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = self.entrada(x)
        x = self.relu(x)

        copia = x

        x = self.res_linear1(x)
        x = self.relu(x)
        x = self.res_linear2(x)

        x = x + copia
        x = self.relu(x)

        x = self.refino(x)
        x = self.relu(x)
        x= self.saida(x)

        return x


class RedeCasasResLN(torch.nn.Module):
    def __init__(self, n_entradas):
        super().__init__()
        self.entrada = torch.nn.Linear(n_entradas, 128)
        self.norm_in = torch.nn.LayerNorm(128)

        self.res_linear1 = torch.nn.Linear(128, 128)
        self.norm_res1 = torch.nn.LayerNorm(128)
        self.res_linear2 = torch.nn.Linear(128, 128)
        self.norm_res2 = torch.nn.LayerNorm(128)

        self.refino = torch.nn.Linear(128, 32)
        self.saida = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.norm_in(self.entrada(x)))

        copia = x
        x = self.relu(self.norm_res1(self.res_linear1(x)))
        x = self.norm_res2(self.res_linear2(x))
        x = self.relu(x + copia)

        x = self.relu(self.refino(x))
        return self.saida(x)

class RedeCasasDrop(torch.nn.Module):
    def __init__(self, n_entradas, p=0.1):
        super().__init__()
        self.linear1 = torch.nn.Linear(n_entradas, 128)
        self.linear2 = torch.nn.Linear(128, 32)
        self.saida = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(p)

    def forward(self, x):
        x = self.drop(self.relu(self.linear1(x)))
        x = self.drop(self.relu(self.linear2(x)))
        return self.saida(x)

class RedeCasasResLNDrop(torch.nn.Module):
    def __init__(self, n_entradas):
        super().__init__()
        self.entrada = torch.nn.Linear(n_entradas, 128)
        self.norm_in = torch.nn.LayerNorm(128)

        self.res_linear1 = torch.nn.Linear(128, 128)
        self.norm_res1 = torch.nn.LayerNorm(128)
        self.res_linear2 = torch.nn.Linear(128, 128)
        self.norm_res2 = torch.nn.LayerNorm(128)

        self.refino = torch.nn.Linear(128, 32)
        self.saida = torch.nn.Linear(32, 1)

        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.1)

    def forward(self, x):
        x = self.drop(self.relu(self.norm_in(self.entrada(x))))

        copia = x
        x = self.relu(self.norm_res1(self.res_linear1(x)))
        x = self.norm_res2(self.res_linear2(x))
        x = self.drop(self.relu(x + copia))

        x = self.relu(self.refino(x))
        return self.saida(x)
