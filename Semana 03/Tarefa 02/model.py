import torch

class RedeCasas(torch.nn.Module):
    def __init__(self, n_entradas):
        super().__init__()

        #camada 1 - 128 neuronios
        self.linear1 = torch.nn.Linear(in_features=n_entradas, out_features=128)
        self.relu1 = torch.nn.ReLU()

        #camada 2 - 64 neuronios
        self.linear2 = torch.nn.Linear(in_features=128, out_features=64)
        self.relu2 = torch.nn.ReLU()

        #saída
        self.saida = torch.nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)

        x = self.linear2(x)
        x = self.relu2(x)

        x = self.saida(x)
        
        return x
