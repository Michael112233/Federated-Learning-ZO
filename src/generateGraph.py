import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fedavg_GD = pd.read_csv("../performance/experiment/rcv/FedAvg_GD/svm/eta=0.001/(time=2023-11-26-16-24-48).csv")
fes1 = np.array(fedavg_GD['current_grad_times'])
loss1 = np.array(fedavg_GD['current_loss'])
plt.semilogy(fes1, loss1, label="fedavg_GD")

fedavg_SGD = pd.read_csv("../performance/experiment/rcv/FedAvg_SGD/svm/eta=0.001/(time=2023-11-26-20-17-17).csv")
fes1 = np.array(fedavg_SGD['current_grad_times'])
loss1 = np.array(fedavg_SGD['current_loss'])
plt.semilogy(fes1, loss1, label="fedavg_SGD")

subspace = pd.read_csv("../performance/experiment/rcv/zeroth_grad/svm/eta=0.001/(time=2023-11-26-15-06-06).csv")
fes2 = np.array(subspace['current_grad_times'])
loss2 = np.array(subspace['current_loss'])
plt.semilogy(fes2, loss2, label="subspace")

fedzo = pd.read_csv("../performance/experiment/rcv/FedZO/svm/eta=0.01/(time=2023-11-26-16-26-57).csv")
fes3 = np.array(fedzo['current_grad_times'])
loss3 = np.array(fedzo['current_loss'])
plt.semilogy(fes3, loss3, color="grey", label="fedzo")

fedsign = pd.read_csv("../performance/experiment/rcv/FedAvg_SignSGD/svm/eta=0.001/(time=2023-11-26-16-25-02).csv")
fes4 = np.array(fedsign['current_grad_times'])
loss4 = np.array(fedsign['current_loss'])
plt.semilogy(fes4, loss4, label="fed_SignSGD")

plt.xlabel("FES")
plt.ylabel("loss")

plt.legend()
plt.show()

