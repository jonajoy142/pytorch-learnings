import torch


print("MPS available:", torch.backends.mps.is_available())
# create tensors
x = torch.tensor([1,2,3])
y = torch.tensor([4,5,6])

print("Tensor x:", x)
print("Tensor y:", y)
print("Addition:", x + y)

# matrix example
a = torch.rand(2,3)
b = torch.rand(3,2)

print("Matrix multiplication:")
print(torch.matmul(a,b))