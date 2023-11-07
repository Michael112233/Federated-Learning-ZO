import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fedavg = pd.read_csv("../performance/experiment/rcv/FedAvg/eta=10/(time=2023-11-07-11-05-37).csv")
fes1 = np.array(fedavg['current_grad_times'])
loss1 = np.array(fedavg['current_loss'])
plt.semilogy(fes1, loss1, label="fedavg")

subspace = pd.read_csv("../performance/experiment/rcv/zeroth_grad/eta=25/(time=2023-11-06-21-51-10).csv")
fes2 = np.array(subspace['current_grad_times'])
loss2 = np.array(subspace['current_loss'])
plt.semilogy(fes2, loss2, label="subspace")

plt.xlabel("FES")
plt.ylabel("loss")

plt.legend()
plt.show()

