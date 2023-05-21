import torch
import torch.nn as nn
import numpy as np


def f(H, L, u, v):
    return (torch.sqrt((H * H + L * L) * u * u - H * H * v * v) - L * v) / (u * u - v * v)


def theta(H, L, u, v):
    return (-H * H * v + L * np.sqrt(4 * (H * H + L * L) * u * u - H * H * v * v)) / (2 * (H * H + L * L) * u)


class T(nn.Module):
    def __init__(self):
        super(T, self).__init__()
        self.L2min = 600 * torch.sqrt(torch.tensor(2.1 * 2.1 - 1.5 * 1.5)) / 1.5
        self.L2 = nn.Parameter(torch.rand([1]) + self.L2min + 2)

    def forward(self):
        ret = f(600, self.L2, 1.5, 2.1) + 2 * f(300, (1000 - self.L2) / 2, 1.5, 0.875)

        return ret

    def theta(self, H, L, u, v):
        return (-H * H * v + L * torch.sqrt(4 * (H * H + L * L) * u * u - H * H * v * v)) / (2 * (H * H + L * L) * u)


if __name__ == '__main__':

    model = T()
    t_min = 10000
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1)
    for i in range(8000):
        model.zero_grad()
        t = model()
        t.backward()
        optimizer.step()
        print(t.data)
        if t_min > t.data:
            t_min = t.data

    print('L2 = ', model.L2.data)
    print('Min Time = ', t_min)

    print([500 - float(model.L2.data[0]) / 2, 300])
    print([500 + float(model.L2.data[0]) / 2, 900])
    # print(theta())
    print('theta1 = ', 180 * np.arccos(theta(300, 500 - float(model.L2.data[0]) / 2, 1.5, 0.875)) / np.pi)
    print('theta2 = ', 180 * np.arccos(theta(600, float(model.L2.data[0]), 1.5, 2.1)) / np.pi)
