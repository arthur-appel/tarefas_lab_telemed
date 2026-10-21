'''no mais, as loss functions que eu fiz já tinham suporte a batches (torch.mean),
então só botei elas aqui mais organizadas e deixei claro no nome da função, e testei com batches'''

import torch

prev_batch=torch.tensor([[172,300,492],[200,150,827]]).float()
real_batch=torch.tensor([[175,312,500],[211,138,816]]).float()

def mse_batches(previstos,reais):
    if previstos.shape != reais.shape:
        raise ValueError("Os tensores devem ter o mesmo shape")
    
    return torch.mean((previstos-reais)**2)
    
def mae_batches(previstos,reais):
    if previstos.shape != reais.shape:
        raise ValueError("Os tensores devem ter o mesmo shape")
    
    return torch.mean(torch.abs(previstos-reais))   



def huber_loss_batches(previstos,reais,delta=1.0):
    if previstos.shape != reais.shape:
        raise ValueError("Os tensores devem ter o mesmo shape")
    
    erro = torch.abs(previstos - reais)
    loss = torch.where(erro <= delta, 0.5 * erro ** 2, delta * (erro - 0.5 * delta))
    
    return torch.mean(loss)


print(huber_loss_batches(prev_batch,real_batch))
print(torch.nn.HuberLoss(delta=1.0)(prev_batch, real_batch)) #pra checagem de valor, deu certinho.

print(mse_batches(prev_batch,real_batch))
print(torch.nn.MSELoss()(prev_batch,real_batch)) #checar valor, deu boa.

print(mae_batches(prev_batch,real_batch))
print(torch.nn.L1Loss()(prev_batch,real_batch)) #deu boa.
