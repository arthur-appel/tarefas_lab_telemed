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
#print(torch.nn.MSELoss()(prev1,real1)) #pra checagem de valor, deu certinho.

print(mae(prev1,real1)) 
#print(torch.nn.L1Loss()(prev1,real1)) #pra checagem de valor, deu certinho.


#observaçoes minhas: a mae pune muito menos erros grandes, usar quando tiver muito outlier.
#mse parece mais sensivel a outliers, talvez usar em situações em que erros grandes não são toleráveis.
#pra resolver a problemática dos outliers, a huber_loss usa um delta para determinar quando usar a mae e quando usar a mse, ou seja, se o erro for menor que delta, usa a mse, se for maior que delta, usa a mae.
#o delta deve ser adaptado de acordo com o problema, se tiver muitos outliers, usar um delta menor, se tiver poucos outliers, usar um delta maior.

def huber_loss(previstos,reais,delta=1.0):
    if previstos.shape != reais.shape:
        raise ValueError("Os tensores devem ter o mesmo shape")
    
    erro = torch.abs(previstos - reais)
    loss = torch.where(erro <= delta, 0.5 * erro ** 2, delta * (erro - 0.5 * delta))
    
    return torch.mean(loss)

huber_prev = torch.tensor([255.25, 512.75, 771.30])
huber_real = torch.tensor([256.8, 580.0, 779.0])

print(huber_loss(huber_prev, huber_real, delta=1.0))
#print(torch.nn.HuberLoss(delta=1.0)(huber_prev, huber_real)) pra checagem de valor, deu certinho.
