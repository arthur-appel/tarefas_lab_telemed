import torch

vetor1= torch.tensor([1, 2, 3])
vetor2= torch.tensor([4, 5, 6])

matriz1= torch.tensor([[1, 2, 3], [4, 5, 6]])
matriz2= torch.tensor([[7, 8, 9], [10, 11, 12]])

g = torch.Generator().manual_seed(42)

tensor_random_1 = torch.randn(3,2)
tensor_random_2 = torch.randn(2,3)

tensor_randint_1 = torch.randint(low=6, high=8, size=(3, 3))
tensor_randint_2 = torch.randint(low=6, high=8, size=(3,3))

resultado = tensor_randint_1 @ tensor_randint_2

print(f"Tensor 1:\n\n{tensor_randint_1}\n\nTensor 2:\n\n{tensor_randint_2}\n\nResultado:\n\n{resultado}")




