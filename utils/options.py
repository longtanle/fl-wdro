#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Mnist", choices=["fiveDigit", "EMNIST", "human_activity", "gleam", "vehicle_sensor", "Mnist", "Synthetic", "Cifar10", "Office_Caltech10"])
    parser.add_argument("--target", type=int, default=-1, help="index of target domain in set of data, choose the last one" )
    parser.add_argument("--model", type=str, default="mclr", choices=["dnn", "mclr", "cnn"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Local learning rate")
    parser.add_argument("--robust", type=int, default=1, help="robust training option")
    parser.add_argument("--gamma", type=float, default=0.1, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--subusers", type=float, default=0.1, help="partition of users")
    parser.add_argument("--algorithm", type=str, default="FedRob",choices=["FedRob","FedAvg"]) 
    parser.add_argument("--numusers", type=int, default=100, help="Total users")
    parser.add_argument("--K", type=int, default=10, help="Computation steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--commet", type=int, default=0, help="log data to commet")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments")
    args = parser.parse_args()
    return args