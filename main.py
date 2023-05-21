# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model3 import Treg
from model3 import train_tr


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    Tmins = np.load('tmins.npy')
    ns = np.arange(9, 119+1)
    plt.plot(ns, Tmins)
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
