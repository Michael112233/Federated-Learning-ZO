import time

from sampling import get_rcv1, get_mnist
from algorithm import FedAvg, Zeroth_grad
from utils import excel_solver, parameter, eta_class, mkdir


def make_dir(dataset_name, algorithm_name, params):
    eta = params.eta
    alpha = params.alpha
    memory_length = params.memory_length

    mkdir("../performance")
    mkdir("../performance/params")

    mkdir("../performance/params/{}/{}".format(dataset_name, algorithm_name))
    mkdir("../performance/params/{}/{}/eta={}".format(dataset_name, algorithm_name, eta))
    if algorithm_name == "zeroth":
        # print(alpha)
        mkdir("../performance/params/{}/{}/eta={}/alpha={:.2}".format(dataset_name, algorithm_name, eta, alpha))
        mkdir(
            "../performance/params/{}/{}/eta={}/alpha={:.2}/memory_length={}".format(dataset_name, algorithm_name, eta,
                                                                                     alpha, memory_length))


def get_result(filename, algorithm):
    csv_solver = excel_solver(filename)
    start_time = time.time()
    current_time, current_grad_times, current_loss, current_round = algorithm.alg_run(start_time)
    # print("{}\n".format(filename))
    csv_solver.save_excel(current_time, current_grad_times, current_loss, current_round)


eta_list = [0.001, 0.01, 0.1, 1, 10]
alpha_list = [0.5]
# eta_list = [100]
dataset_list = ['mnist', 'rcv']
algorithm_list = ['FedAvg', 'zeroth']
# dataset_list = ['rcv']
# algorithm_list = ['zeroth']
memory_length_list = [5]
times = 3
verbose = True
batch_size = 1000

eta_choose = eta_class()
eta_type = eta_choose.choose(2)

for dataset_name in dataset_list:
    mkdir("../performance/params/{}".format(dataset_name))
    for algorithm_name in algorithm_list:
        for eta in eta_list:
            # for dataset
            if dataset_name == 'rcv':
                dataset, X, Y, global_model = get_rcv1()
            else:
                dataset, X, Y, global_model = get_mnist()
            max_grad_time = 300 * dataset.length()

            # for algorithm
            if algorithm_name == 'FedAvg':
                for i in range(times):
                    filename = "../performance/params/{}/{}/eta={}/({}).csv".format(
                        dataset_name, algorithm_name, eta, i + 1)
                    print(filename)
                    if dataset_name == "mnist":
                        para = parameter(max_grad_time, eta_type, eta, 0, 0, 1000, 10, verbose)
                    else:
                        para = parameter(max_grad_time, eta_type, eta, 0, 0, 1000, 100, verbose)
                    make_dir(dataset_name, algorithm_name, para)
                    algorithm = FedAvg(dataset, global_model, para)
                    get_result(filename, algorithm)
            elif algorithm_name == 'zeroth':
                for alpha in alpha_list:
                    for memory_length in memory_length_list:

                        for i in range(times):
                            filename = "../performance/params/{}/{}/eta={}/alpha={:.2}/memory_length={}/({}).csv".format(
                                dataset_name,
                                algorithm_name, eta, alpha,
                                memory_length,
                                i + 1)
                            if dataset_name == "mnist":
                                para = parameter(max_grad_time, eta_type, eta, alpha, memory_length, 1000, 10, verbose)
                            else:
                                para = parameter(max_grad_time, eta_type, eta, alpha, memory_length, 1000, 100, verbose)
                            make_dir(dataset_name, algorithm_name, para)
                            print(filename)
                            algorithm = Zeroth_grad(dataset, global_model, para)
                            get_result(filename, algorithm)
