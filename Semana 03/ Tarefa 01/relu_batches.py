#learning rate - de maneira resumida, é o tamanho do passo que o modelo irá dar na hora de corrigir os pesos, passos pequenos fazem o treinamento do modelo demorar demais, passos muito grandes podem piorar o modelo.
#otimizador - pega a info do erro, vinda do backpropagation, decide a mecânica de como os pesos vão ser ajustados. ele pode adaptar a learning rate a depender do otimizador.
#SGD - olha para a inclincação e dá um passo de tamanho fixo, pode ficar passando e voltando do ponto mínimo de erro, precisa ser bem calibrado.
#ADAM - se baseia em momentum e adapta o tamanho dos passos com base na mudança dos pesos. 
#o batch é um lote, ou recorte de dados que você irá utilizar para treinar seu modelo por partes, e no final tirar a média dos erros desses batches, e assim entender o comportamento geral da base de dados, mas sem precisar processar ela toda de uma vez.
# é uma funçao responsável por quebrar a limitação da linearidade que temos ao somar nossas variáveis, pesos e biases, o que permite que a rede aprenda padrões mais complexos e não lineares.

import torch

tensor1= torch.tensor([1,2,-3])
tensor2= torch.tensor([4,-7,1])

batch1=torch.stack([tensor1,tensor2])



def relu_batches(amostra):
    return torch.maximum(amostra,torch.tensor(0.0))

print(relu_batches(tensor1)) #testando só com um tensor

print(relu_batches(batch1)) #testando com batch
