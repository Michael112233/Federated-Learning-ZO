import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fedavg_GD = pd.read_csv("../performance/experiment/rcv/FedAvg_GD/eta=10/(time=2023-11-07-22-37-57).csv")
fes1 = np.array(fedavg_GD['current_grad_times'])
loss1 = np.array(fedavg_GD['current_loss'])
plt.semilogy(fes1, loss1, label="fedavg_GD")

fedavg_SGD = pd.read_csv("../performance/experiment/rcv/FedAvg_SGD/eta=10/(time=2023-11-08-17-40-21).csv")
fes3 = np.array(fedavg_SGD['current_grad_times'])
loss3 = np.array(fedavg_SGD['current_loss'])
plt.semilogy(fes3, loss3, label="fedavg_SGD")

subspace = pd.read_csv("../performance/experiment/rcv/zeroth_grad/eta=25/(time=2023-11-08-11-15-57).csv")
fes2 = np.array(subspace['current_grad_times'])
loss2 = np.array(subspace['current_loss'])
plt.semilogy(fes2, loss2, label="subspace")

fedzo = pd.read_csv("../performance/experiment/rcv/FedZO/eta=50/(time=2023-11-10-22-43-53).csv")
fes3 = np.array(fedzo['current_grad_times'])
loss3 = np.array(fedzo['current_loss'])
plt.semilogy(fes3, loss3, color="grey", label="fedzo")

plt.xlabel("FES")
plt.ylabel("loss")

plt.legend()
plt.show()

