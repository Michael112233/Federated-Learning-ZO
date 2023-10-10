import time

from sampling import get_rcv1, get_mnist
from algorithm import FedAvg, Zeroth_grad
from utils import excel_solver, parameter, eta_class

eta_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
alpha_list = [0.05 * i for i in range(1, 21)]
dataset_list = ['mnist', 'rcv']
algorithm_list = ['FedAvg', 'zeroth']
memory_length_list = [5, 10, 15, 20]
times = 3
verbose = False

eta_choose = eta_class()
eta_type = eta_choose.choose(2)

for dataset_name in dataset_list:
    if dataset_name == 'rcv':
        dataset, X, Y, global_model = get_rcv1()
    else:
        dataset, X, Y, global_model = get_mnist()
    max_grad_time = 5000 * dataset.length()
    batch_size = 1000

    for alpha in alpha_list:
        for eta in eta_list:
            for memory_length in memory_length_list:
                for algorithm_name in algorithm_list:
                    para = parameter(max_grad_time, eta_type, eta, alpha, memory_length, batch_size, verbose)
                    if algorithm_name == 'zeroth_grad':
                        algorithm = Zeroth_grad(dataset, global_model, para)
                    else:
                        algorithm = FedAvg(dataset, global_model, para)

                    for i in range(times):
                        filename = "../performance/params/" + dataset_name + "/{}/alpha={:.2}/eta={}/memory_length={}/({}).csv".format(algorithm_name, alpha, eta, memory_length,i+1)
                        csv_solver = excel_solver(filename)

                        start_time = time.time()
                        current_time, current_grad_times, current_loss, current_round = algorithm.alg_run(start_time)
                        print("The loss is {} and The eta is {}".format(current_loss[-1], eta))
                        csv_solver.save_excel(current_time, current_grad_times, current_loss, current_round)
