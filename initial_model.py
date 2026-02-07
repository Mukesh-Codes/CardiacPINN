import torch
import torch.nn as nn
from torchdiffeq import odeint

#The neural ODE class. This is used in the PINN.
class NeuralODE(nn.Module):
    def __init__(self):
        super(NeuralODE, self).__init__()
        layers = [
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2)
        ] #just for trying out.
        self.net = nn.Sequential(*layers)

    def forward(self, t, y):
        return self.net(y)

    def ode_solve(self, y0, t):
        return odeint(self, y0, t)

    def compute_loss(self, y0, t, y_true):
        y_pred = self.ode_solve(y0, t)
        loss = nn.MSELoss()(y_pred, y_true)
        return loss

if __name__ == "__main__":
    y_true = torch.tensor([[1.0, 0.0],
                            [0.0, 1.0],
                            [-1.0, 0.0],
                            [0.0, -1.0]])
    t = torch.tensor([0.0, 1.0, 2.0, 3.0])
    y0 = torch.tensor([1.0, 0.0])
    model = NeuralODE()
    for epoch in range(100):
        loss= model.compute_loss(y0, t, y_true)
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        loss.backward()
