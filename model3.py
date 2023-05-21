import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def f(H, L, u, v):
    return (torch.sqrt((H * H + L * L) * u * u - H * H * v * v) - L * v) / (u * u - v * v)


def v(y):
    return 2.1 * y * (1200 - y) / 360000


class Treg(nn.Module):
    def __init__(self, n, H, L, u, v):
        super(Treg, self).__init__()
        self.n = n
        self.H = H
        self.L = L
        self.u = u
        self.stream_velocity = v
        self.Ls = nn.Parameter(torch.ones([n]) / (n + 1) * L / 2)
        self.dH = torch.tensor(H / (n + 1))
        self.L_f = torch.tensor([0])
        self.ys = torch.linspace(0, H / 2, n + 1)
        self.vs = self.stream_velocity(self.ys)
        self.Tmin = 0

    def forward(self):
        self.L_f = self.L / 2 - torch.sum(self.Ls)
        return 2 * (torch.sum(f(self.dH, self.Ls, self.u, self.vs[0:-1])) + f(self.dH, self.L_f, self.u, self.vs[-1]))

    def route(self):
        ys = np.linspace(0, self.H, self.n * 2 + 2)
        xs = np.zeros(shape=[self.n * 2 + 2])
        x = 0
        for i in range(0, self.n):
            x += self.Ls[i]
            xs[i + 1] = x

        xs[2 * self.n + 1:self.n:-1] = self.L - xs[0:self.n + 1]

        return np.array([xs, ys])

    def save_route(self, path):
        data = pd.DataFrame(self.route().transpose())

        writer = pd.ExcelWriter(path)
        data.to_excel(writer, 'page_1', float_format='%.5f')
        writer.save()

    def thetas(self):
        u = self.u
        ret = torch.zeros([2 * self.n + 2])
        ret[0:self.n] = ((-self.dH * self.dH * self.vs[0:-1] +
                          self.Ls * torch.sqrt(
                    4 * (self.dH * self.dH + self.Ls * self.Ls) * u * u - self.dH * self.dH * self.vs[0:-1]
                    * self.vs[0:-1]))
                         / (2 * (self.dH * self.dH + self.Ls * self.Ls) * u))
        ret[self.n] = ((-self.dH * self.dH * self.vs[-1] +
                        self.L_f * torch.sqrt(4 * (self.dH * self.dH + self.L_f * self.L_f) * u * u -
                                              self.dH * self.dH * self.vs[-1] * self.vs[-1]))
                       / (2 * (self.dH * self.dH + self.L_f * self.L_f) * u))
        ret = np.array(ret.detach().numpy())
        ret[2 * self.n + 1:self.n:-1] = ret[0:self.n + 1]
        ret = np.arccos(np.abs(ret)) * (ret >= 0) - np.arccos(np.abs(ret)) * (ret < 0)
        ret += (ret < 0) * np.pi
        ret[self.n] = ret[self.n - 1]
        ret[self.n + 1] = ret[self.n + 2]
        return ret


def train_tr(tr, opt, epoch, save=False):
    t_min = 10000
    for i in range(epoch):
        tr.zero_grad()
        t = tr()
        t.backward()
        opt.step()
        # print(t.data)
        if t_min > t.data:
            t_min = t.data

    tr.Tmin = t_min
    if save:
        torch.save(treg, './model3-5.pt')

    print('Min Time = ', t_min)


if __name__ == '__main__':
    treg = Treg(99, 1200, 1000, 2, v)
    # treg = torch.load('./model3.pt')
    optimizer = torch.optim.Adam(params=treg.parameters(), lr=0.01)
    train_tr(treg, optimizer, 8000)
    print(treg.n)

    # print(treg.route())
    route = treg.route()
    # treg.save_route('route.xlsx')
    # print(route)
    plt.plot(route[0, :], route[1, :])
    plt.show()
    thetas = treg.thetas()
    ys = np.linspace(0, 1200, treg.n * 2 + 2)
    plt.plot(ys, thetas/np.pi*180)
    plt.show()
