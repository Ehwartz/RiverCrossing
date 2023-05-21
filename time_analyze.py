import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model3 import Treg
from model3 import train_tr


def v(y):
    return 2.1 * y * (1200 - y) / 360000


def generate_trs(start: int, end: int):
    trs = list()
    for i in range(end - start + 1):
        trs.append(Treg(i + start, 1200, 1000, 2, v))
    return trs


def train_trs(trs, epoch):
    for i in range(len(trs)):
        print(i)
        optimizer = torch.optim.Adam(params=trs[i].parameters(), lr=0.01)
        train_tr(trs[i], optimizer, epoch)


def get_Tmin(trs):
    Tmins = np.empty(shape=[len(trs)])
    for i in range(len(trs)):
        Tmins[i] = trs[i].Tmin
    return Tmins


if __name__ == '__main__':
    start = 9
    end = 119
    trs = generate_trs(start, end)
    train_trs(trs, 2000)
    Tmins = get_Tmin(trs)
    ns = np.arange(start, end+1)
    plt.plot(ns, Tmins)
    plt.show()

    np.save('tmins.npy', Tmins)