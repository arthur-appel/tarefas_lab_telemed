'''
conceitos:

- backpropagation: ajuda a determinar o quanto cada peso da rede neural contribuiu para o erro final
, o algoritmo usa como base as derivadas parciais da loss function em relação a cada peso da rede neural,
e a partir disso é possível ajustar os pesos da rede neural para minimizar o erro.

- loss function: é uma função que mede o quão bem a rede neural tá performando, ou seja, mede o erro entre os valores previstos e os valores reais.
'''

import torch

def mse(previstos,reais):
    if previstos.shape != reais.shape:
        raise ValueError("Os tensores devem ter o mesmo shape")
    
    return torch.mean((previstos-reais)**2)
    
def mae(previstos,reais):
    if previstos.shape != reais.shape:
        raise ValueError("Os tensores devem ter o mesmo shape")
    
    return torch.mean(torch.abs(previstos-reais))   

prev1 = torch.tensor([255.25,512.75,771.30])
real1 = torch.tensor([256.8, 580.0, 779.0])

print(mse(prev1,real1))
print(mae(prev1,real1)) 

#observaçoes minhas: a mae pune muito menos erros grandes, usar quando tiver muito outlier.
#mse parece mais sensivel a outliers, talvez usar em situações em que erros grandes não são toleráveis.



