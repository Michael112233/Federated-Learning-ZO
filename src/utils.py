#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

def get_loss(global_model, dataset, weights, current_round, losses):
    Xfull, Yfull = dataset.full()
    l = global_model.loss(weights, Xfull, Yfull)
    acc = global_model.acc(weights, Xfull, Yfull)
    print("After iteration {}: loss is {} and accuracy is {:.2f}%".format(current_round, l, acc))
    losses.append(l)
    return losses
    # print("After iteration {}: loss is {}".format(i + 1, l))


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
