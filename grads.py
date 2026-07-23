import torch as pt
import torch.nn.functional as F
import matplotlib.pyplot as plt

class NET(pt.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = pt.nn.Linear(2, 1)
        pt.nn.init.ones_(self.layer1.weight)
        pt.nn.init.zeros_(self.layer1.bias)

    def forward(self, x):
        x = self.layer1(x)
        return(F.sigmoid(x)) # need an activation with a second derivative

N_POINTS = 2
x = pt.linspace(0., 1., N_POINTS)
y = pt.linspace(0., 1., N_POINTS)
data = pt.cartesian_prod(x, y) # like meshgrid, but gives a vector
data.requires_grad = True
print(f"{data=}")

model = NET()
optimizer = pt.optim.SGD(model.parameters())

for p in model.parameters():
    print(f"{p.grad=}")

u = model(data)
print(f"{u=}")

print(f"{data.grad=}")
du = pt.autograd.grad(u, data, grad_outputs=pt.ones(u.shape), create_graph=True)[0]
print(f"{data.grad=}") # no change in data.grad - output only goes to du
print(f"{du=}")

d2u = pt.autograd.grad(du, data, grad_outputs=pt.ones(du.shape), create_graph=True)[0]
print(f"{d2u=}")

laplacian = d2u.sum(dim=1)
print(laplacian)

mse = pt.nn.MSELoss()
loss = mse(laplacian, pt.zeros(N_POINTS**2))
loss.backward()
print(f"{data.grad=}") # be aware data.grad changes via the .backward() call

optimizer.step() # only alters the model parameter values
optimizer.zero_grad() # only zeros gradients of model parameters

u = model(data)

du = pt.autograd.grad(u, data, grad_outputs=pt.ones(u.shape), create_graph=True)[0]
print(f"{du=}")
