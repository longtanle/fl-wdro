import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy

class User:
    """
    Base class for users in federated learning.
    """
    def __init__(self, device, id, train_data, test_data, model, batch_size = 0, learning_rate = 0, robust = 0 , gamma = 0, local_epochs = 0):
        # from fedprox
        self.device = device
        self.model = copy.deepcopy(model)
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        #print("Len train and test",len(train_data),len(test_data))
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.robust = robust
        self.gamma = gamma
        self.local_epochs = local_epochs
        self.target = False

        if(self.batch_size == 0):
            self.trainloader = DataLoader(train_data, self.train_samples,shuffle=True)
            self.testloader =  DataLoader(test_data, self.test_samples,shuffle=True)
        else:
            self.trainloader = DataLoader(train_data, self.batch_size,shuffle=True)
            self.testloader =  DataLoader(test_data, self.batch_size,shuffle=True)

        self.testloaderfull = DataLoader(test_data, self.test_samples,shuffle=True)
        self.trainloaderfull = DataLoader(train_data, self.train_samples,shuffle=True)

        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

        # those parameters are for persionalized federated learing.
        #self.local_model = copy.deepcopy(list(self.model.parameters()))

    def set_target(self):
        self.target = True

    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()
            #local_param.data = new_param.data.clone()

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()
    
    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param
    
    def get_updated_parameters(self):
        return self.local_weight_updated
    
    def update_parameters(self, new_params):
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def test(self):
        self.model.eval()
        test_acc = 0
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            #x , y = self.perturb(x, y)
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        return test_acc, y.shape[0]

    def test_robust(self, attack_mode = 'pgd'):
        self.model.eval()
        test_acc = 0
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            if(attack_mode == 'pgd'):
                x = self.pgd_linf(X = x, y = y)
            elif(attack_mode == 'fgsm'):
                x = self.fgsm(X = x, y = y)
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        return test_acc, y.shape[0]
    
    def test_domain(self):
        self.model.eval()
        test_acc = 0
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        return test_acc / y.shape[0]

    def train_error_and_loss(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y.long())
            #print(self.id + ", Train Accuracy:", train_acc)
            #print(self.id + ", Train Loss:", loss)
        return train_acc, loss , self.train_samples
     
    def get_next_train_batch(self):
        if(self.batch_size == 0):
            for X, y in self.trainloaderfull:
                return (X.to(self.device), y.to(self.device))
        else:
            try:
                # Samples a new batch for persionalizing
                (X, y) = next(self.iter_trainloader)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                self.iter_trainloader = iter(self.trainloader)
                (X, y) = next(self.iter_trainloader)
            return (X.to(self.device), y.long().to(self.device))
    
    def get_next_test_batch(self):
        if(self.batch_size == 0):
            for X, y in self.testloaderfull:
                return (X.to(self.device), y.to(self.device))
        else:
            try:
                # Samples a new batch for persionalizing
                (X, y) = next(self.iter_testloader)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                self.iter_testloader = iter(self.testloader)
                (X, y) = next(self.iter_testloader)
            return (X.to(self.device), y.long().to(self.device))
        
    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))

    def fgsm(self, X, y, epsilon):
        """ Construct FGSM adversarial examples on the examples X"""
        delta = torch.zeros_like(X, requires_grad=True)
        loss = self.loss(self.model(X + delta), y)
        loss.backward()
        return X + epsilon * delta.grad.detach().sign()
        
    def pgd_linf(self, X, y, epsilon = 0.3, alpha = 0.01, num_iter = 10):
        ' Construct FGSM adversarial examples on the examples X'
        delta = torch.zeros_like(X, requires_grad=True).to(self.device)
        for t in range(num_iter):
            loss = self.loss(self.model(X + delta), y)
            loss.backward()
            sign = delta.grad.detach().sign()
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.grad.zero_()
        temp = delta.detach()
        return X + temp

    def wasssertein(self, X, y, epsilon = 0.0005, alpha = 0.01, num_iter = 20):
        ' Construct FGSM adversarial examples on the examples X'
        X_adv = X.clone() + torch.rand(X.shape).clamp(-epsilon,epsilon).to(self.device)
        X_adv.requires_grad_(True)
        for t in range(num_iter):
            loss1 = self.loss(self.model(X_adv), y)
            loss2 = 0.5 * self.gamma * torch.norm(X_adv - X)**2 / len(X_adv)
            loss = loss1 - loss2
            loss.backward()
            X_adv.data = (X_adv.data + len(X_adv) * X_adv.grad)
            #delta = X_adv - X
            #norm_delta = torch.norm(delta)
            #norm_grad = torch.norm(X_adv.grad)
            if(torch.norm(X_adv.grad) < 1e-4):
                break
            X_adv.grad.zero_()
        return X_adv

    def pgd_l2(self, X, y, epsilon, alpha, num_iter):
        delta = torch.zeros_like(X, requires_grad=True)
        for t in range(num_iter):
            loss = self.loss(self.model(X + delta), y)
            loss.backward()
            delta.data += alpha*delta.grad.detach() / self.norms(delta.grad.detach())
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
            delta.data *= epsilon / self.norms(delta.detach()).clamp(min=epsilon)
            delta.grad.zero_()
            
        return delta.detach()

    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))
