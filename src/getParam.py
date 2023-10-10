
eta_list = [0.001, 0.01, 0.1, 1, 10, 100]
alpha_list = [0.05 * i for i in range(1, 21)]
dataset_list = ['mnist', 'rcv']
times = 3

for dataset in dataset_list:
    for alpha in alpha_list:
        for eta in eta_list:
            for i in range(times):
                filename = "/performance/params/" + dataset + "/alpha={:.2}, eta={}({}).csv".format(alpha, eta, i+1)
                print(filename)