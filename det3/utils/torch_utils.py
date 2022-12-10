'''
File Created: Wednesday, 17th July 2019 4:49:47 pm

'''
import sys
import torch
from torch import nn
import matplotlib.pyplot as plt
import pickle

class GradientLogger:
    '''
    Log the gradient for torch model
    '''
    def __init__(self):
        self.data = dict()
        self.model = None
        self.plot_ylim = None

    def set_model(self, model: nn.Module):
        self.model = model
        l = [module for module in self.model.modules() if list(module.children()) == []]
        for i, layer in enumerate(l):
            if not hasattr(layer, "weight"):
                continue
            layer_name = layer._get_name()+"_"+str(i)
            layer.weight.register_hook(self._get_grad(layer_name))

    def _get_grad(self, name):
        def hook(grad):
            if name not in self.data.keys():
                self.data[name] = [torch.mean(grad.detach()),
                                   torch.std(grad.detach()),
                                   1]
            else:
                self.data[name][0] = self.data[name][0] * self.data[name][2] + torch.mean(grad.detach())
                self.data[name][1] = self.data[name][1] * self.data[name][2] + torch.std(grad.detach())
                self.data[name][2] += 1
                self.data[name][0] /= self.data[name][2]
                self.data[name][1] /= self.data[name][2]
        return hook

    def log(self, epoch):
        if self.model is None:
            return dict()
        output = dict()
        for k, v in self.data.items():
            output[k] = [itm.cpu().numpy() for itm in v[:-1]]
        output["epoch"] = epoch
        self.data = dict()
        return output

    def plot(self, grad_dict, path, ylim=None):
        if grad_dict == dict():
            return
        name_list = []
        mean_list = []
        std_list = []
        std_list1 = []
        std_list2 = []
        for k, vs in grad_dict.items():
            if k == "epoch":
                continue
            name_list.insert(0, k) # insert at the beginning, otherwise the list will be reversed
            mean_list.insert(0, vs[0])
            std_list1.insert(0, vs[0] - vs[1]/0.5)
            std_list2.insert(0, vs[0] + vs[1]/0.5)
        plt.figure(figsize=(20, 10))
        plt.bar(name_list, mean_list)
        plt.fill_between(name_list, std_list1, std_list2, alpha=0.2)
        plt.xticks(rotation=90)
        plt.grid("on")
        if ylim is not None:
            self.plot_ylim = ylim
        elif self.plot_ylim is None:
            self.plot_ylim = plt.gca().get_ylim()
        else:
            pass
        plt.ylim(self.plot_ylim)
        if "epoch" in grad_dict:
            plt.title("gradient_"+str(grad_dict["epoch"]))
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    def save_pkl(self, grad_dict, path):
        if grad_dict == dict():
            return
        with open(path, 'wb') as f:
            pickle.dump(grad_dict, f, pickle.HIGHEST_PROTOCOL)

    def load_pkl(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

class ActivationLogger:
    '''
    Log the activation value for torch model
    '''
    def __init__(self):
        self.data = dict()
        self.model = None
        self.plot_ylim = None

    def set_model(self, model: nn.Module):
        self.model = model
        l = [module for module in self.model.modules() if list(module.children()) == []]
        for i, layer in enumerate(l):
            layer_name = layer._get_name()+"_"+str(i)
            layer.register_forward_hook(self._get_activation(layer_name))

    def _get_activation(self, name):
        def hook(grad, input, output):
            if type(output).__name__ == "tuple":
                data = output[0].detach()
            elif type(output).__name__ == "SparseConvTensor":
                data = output.features.detach()
            elif type(output).__name__ == "Tensor":
                data = output.detach()
            else:
                raise RuntimeError
            if name not in self.data.keys():
                self.data[name] = [torch.mean(data),
                                   torch.std(data),
                                   1]
            else:
                self.data[name][0] = self.data[name][0] * self.data[name][2] + torch.mean(data)
                self.data[name][1] = self.data[name][1] * self.data[name][2] + torch.std(data)
                self.data[name][2] += 1
                self.data[name][0] /= self.data[name][2]
                self.data[name][1] /= self.data[name][2]
        return hook

    def log(self, epoch):
        if self.model is None:
            return dict()
        output = dict()
        for k, v in self.data.items():
            output[k] = [itm.cpu().numpy() for itm in v[:-1]]
        output["epoch"] = epoch
        self.data = dict()
        return output

    def plot(self, grad_dict, path, ylim=None):
        if grad_dict == dict():
            return
        name_list = []
        mean_list = []
        std_list = []
        std_list1 = []
        std_list2 = []
        for k, vs in grad_dict.items():
            if k == "epoch":
                continue
            name_list.append(k) # insert at the beginning, otherwise the list will be reversed
            mean_list.append(vs[0])
            std_list1.append(vs[0] - vs[1]/0.5)
            std_list2.append(vs[0] + vs[1]/0.5)
        plt.figure(figsize=(20, 10))
        plt.bar(name_list, mean_list)
        plt.fill_between(name_list, std_list1, std_list2, alpha=0.2)
        plt.xticks(rotation=90)
        plt.grid("on")
        if ylim is not None:
            self.plot_ylim = ylim
        elif self.plot_ylim is None:
            self.plot_ylim = plt.gca().get_ylim()
        else:
            pass
        plt.ylim(self.plot_ylim)
        if "epoch" in grad_dict:
            plt.title("activation_"+str(grad_dict["epoch"]))
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    def save_pkl(self, grad_dict, path):
        if grad_dict == dict():
            return
        with open(path, 'wb') as f:
            pickle.dump(grad_dict, f, pickle.HIGHEST_PROTOCOL)

    def load_pkl(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)