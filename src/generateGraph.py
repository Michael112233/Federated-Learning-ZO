import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ./performance/experiment/rcv/FedAvg_GD/logistic/eta=10/(time=2023-11-27-23-32-30).csv
fedavg_GD = pd.read_csv("../performance/experiment/rcv/FedAvg_GD/logistic/eta=10/(time=2023-11-27-23-32-30).csv")
fes1 = np.array(fedavg_GD['current_grad_times'])
loss1 = np.array(fedavg_GD['current_loss'])
fes1[0] += 1
plt.semilogx(fes1, loss1, label="fedavg_GD")

# ./performance/experiment/rcv/FedAvg_SGD/logistic/eta=10/(time=2023-11-27-23-32-04).csv
fedavg_SGD = pd.read_csv("../performance/experiment/rcv/FedAvg_SGD/logistic/eta=10/(time=2023-11-27-23-32-04).csv")
fes5 = np.array(fedavg_SGD['current_grad_times'])
loss5 = np.array(fedavg_SGD['current_loss'])
fes5[0] += 1
plt.semilogx(fes5, loss5, label="fedavg_SGD")

# ./performance/experiment/rcv/zeroth_grad/logistic/eta=25/(time=2023-11-27-23-31-38).csv
subspace = pd.read_csv("../performance/experiment/rcv/zeroth_grad/logistic/eta=25/(time=2023-11-27-23-31-38).csv")
fes2 = np.array(subspace['current_grad_times'])
loss2 = np.array(subspace['current_loss'])
fes2[0] += 1
plt.semilogx(fes2, loss2, label="subspace")

# ./performance/experiment/rcv/FedZO/logistic/eta=50/(time=2023-11-27-23-32-52).csv
fedzo = pd.read_csv("../performance/experiment/rcv/FedZO/logistic/eta=50/(time=2023-11-27-23-32-52).csv")
fes3 = np.array(fedzo['current_grad_times'])
loss3 = np.array(fedzo['current_loss'])
fes3[0] += 1
plt.semilogx(fes3, loss3, color="grey", label="fedzo")

# ../performance/experiment/rcv/FedAvg_SignSGD/logistic/eta=0.01/(time=2023-11-27-23-31-18).csv
fedsign = pd.read_csv("../performance/experiment/rcv/FedAvg_SignSGD/logistic/eta=0.01/(time=2023-11-27-23-31-18).csv")
fes4 = np.array(fedsign['current_grad_times'])
loss4 = np.array(fedsign['current_loss'])
fes4[0] += 1
plt.semilogx(fes4, loss4, label="fed_SignSGD")

plt.xlabel("FES")
plt.ylabel("loss")
plt.title("rcv")

plt.xlim(1, 1e9)

plt.legend()
plt.show()

