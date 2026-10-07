import torch

def multiplicar_tensores(shape_a: tuple,shape_b: tuple):

    if shape_a[-1] != shape_b[-2]:
        raise ValueError("A matriz A deve ter o n de colunas igual o n de linhas da matriz B")

    tensor_a = torch.randn(shape_a)
    tensor_b = torch.randn(shape_b)

    return tensor_a @ tensor_b


shape_1=(2,3)
shape_2=(3,4)
resultado = multiplicar_tensores(shape_1,shape_2)

print("Shape resultante:", resultado.shape)
print("Resultado:\n", resultado)

