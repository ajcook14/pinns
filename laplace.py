# pinn for solving the laplace equation

import torch as pt
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np
from matplotlib.ticker import LinearLocator


class NET(pt.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = pt.nn.Linear(2, 100)
        self.layer2 = pt.nn.Linear(100, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = F.sigmoid(x)
        x = self.layer2(x)
        x = F.sigmoid(x)
        return(x) # need an activation with a second derivative

N_POINTS = 20
lbound = 0.
ubound = 1.
x = pt.linspace(lbound, ubound, N_POINTS)
y = pt.linspace(lbound, ubound, N_POINTS)
data = pt.cartesian_prod(x, y) # like meshgrid, but gives a vector
data.requires_grad = True

mask = (
    (data[:, 0] == lbound) | (data[:, 0] == ubound) |
    (data[:, 1] == lbound) | (data[:, 1] == ubound)
)

with pt.no_grad():
    boundary = data[mask]

model = NET()
optimizer = pt.optim.SGD(model.parameters(), lr=0.1)

print("training...")
for i in range(500):
    optimizer.zero_grad() # only zeros gradients of model parameters
    u = model(data)

    du = pt.autograd.grad(u, data, grad_outputs=pt.ones(u.shape), create_graph=True)[0]
    # no change in data.grad - output only goes to du

    d2u = pt.autograd.grad(du, data, grad_outputs=pt.ones(du.shape), create_graph=True)[0]

    laplacian = d2u.sum(dim=1)

    with pt.no_grad():
        dirichlet = boundary[:, 0]**2 - boundary[:, 1]**2 # real part of z -> z^2
    
    u_boundary = model(boundary).flatten()

    mse = pt.nn.MSELoss()
    loss = mse(laplacian, pt.zeros(N_POINTS**2)) + mse(u_boundary, dirichlet)
    loss.backward() # be aware data.grad changes via the .backward() call

    optimizer.step() # only alters the model parameter values

with pt.no_grad():
    U = model(data).numpy()

U = U.reshape((N_POINTS, N_POINTS))
X, Y = np.meshgrid(x.numpy(), y.numpy())

ax = plt.figure().add_subplot(projection='3d')

colortuple = ('y', 'b')
colors = np.empty((N_POINTS, N_POINTS), dtype=str)

for y in range(N_POINTS):
    for x in range(N_POINTS):
        colors[y, x] = colortuple[(x + y) % len(colortuple)]

# Plot the surface with face colors taken from the array we made.
surf = ax.plot_surface(X, Y, U, facecolors=colors, linewidth=0)

# Customize the z axis.
ax.set_zlim(0, 1)
ax.zaxis.set_major_locator(LinearLocator(6))

plt.show()