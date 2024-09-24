import torch
import torch.nn as nn
import yaml
import torch.optim as optimis
with open("Config.yaml", "r") as f:
    config = yaml.safe_load(f)


def get_optimizer(config, params):
    params_dict = config["optim"]["params"]
    algo = config["optim"]["algo"]
    optimizer = eval(f"torch.optim.{algo}(params, **params_dict)")
    return optimizer
def get_scheduler(optimizer):
    return ( torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1))

def loss_function(config):

    loss_fct = config["training"]["loss"]
    loss = eval(f"nn.{loss_fct}()")
    return loss


if __name__ == "__main__":

    model_parameters_exemple = [
        torch.nn.Parameter(torch.Tensor([[1.0, 2.0], [3.0, 4.0]])),
        torch.nn.Parameter(torch.Tensor([5.0, 6.0])),
        torch.nn.Parameter(torch.Tensor([7.0])),
    ]

    optimizer_test = get_optimizer(config, model_parameters_exemple)
    loss_test = loss_function(config)
