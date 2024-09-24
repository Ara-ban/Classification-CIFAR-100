import os
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix



def train(config, model, loss_function, optimizer,scheduler, device, data_loader):
    """
    this training function is over one epoch :
    Arguments:

    model         -- torch.nn.Module object
    loss_function -- a loss module
    optmizer      -- torch.optim.Optimzer object
    device        -- torch.device object
    data_loader   -- torch.utils.data.DataLoader object

    Returns:

     total loss  --  float



    """

    model.train()
    total_loss = 0
    num_samples = 0

    for input_tensor, target in data_loader:
        input_tensor, target = input_tensor.to(device), target.to(device)

        # Compute the forward propagation
        y_hat = model(input_tensor)
        loss = loss_function(y_hat, target)

        llambda = config["regularisation"]["coef"]
        type = config["regularisation"]["type"]
        if type == "Lasso":
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + llambda * l1_norm
        if type == "Ridge":
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + llambda * l2_norm
        else:
            loss = loss

        # Compute of the backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update of the loss
        total_loss += input_tensor.shape[0] * loss.item()
        num_samples += input_tensor.shape[0]
    if config["optim"]["scheduler"]:
    	scheduler.step()
    total_loss = total_loss / num_samples
    return total_loss



def test(model, loss_function, device, data_loader):
    """

    Arguments:

    model         -- torch.nn.Module object
    loss_function -- a loss module
    device        -- torch.device object
    data_loader   -- torch.utils.data.DataLoader object

    Returns:

     total loss  --  float



    """

    model.eval()
    total_loss = 0
    num_samples = 0
    correct = 0
    for input_tensor, target in data_loader:
        input_tensor, target = input_tensor.to(device), target.to(device)

        y_hat = model(input_tensor)
        loss = loss_function(y_hat, target)

        # Update of the loss
        total_loss += input_tensor.shape[0] * loss.item()
        num_samples += input_tensor.shape[0]

        predicted_targets = y_hat.argmax(dim=1)

        correct += (predicted_targets == target).sum().item()

    accuracy = correct/num_samples

    total_loss = total_loss / num_samples

    return total_loss, accuracy



def mean_std(loader):
    # Compute the mean over minibatches
    mean = 0
    for imgs, _ in loader:
        mean += imgs.sum(dim=0)
    mean = mean / len(loader.dataset)

    # Compute the std over minibatches
    std = torch.zeros_like(mean)
    for imgs, _ in loader:
        std += ((imgs - mean) ** 2).sum(dim=0)
    std = std / len(loader.dataset)
    std = torch.sqrt(std)

    # Set the variance of pixels with no variance to 1
    # Because there is no variance
    # these pixels will anyway have no impact on the final decision
    std[std == 0] = 1

    return mean, std


class ModelCheckpoint:
    def __init__(self, model, filepath):
        self.min_loss = 1E12
        self.model = model
        self.filepath = filepath

    def update(self, loss):
        if self.min_loss == 1E12 or loss < self.min_loss:
            self.min_loss = loss
            torch.save(self.model.state_dict(), self.filepath)
            print("saving a better model")
            self.min_loss = loss


def generate_unique_logpath(logdir, raw_run_name):
    """
    Generate a unique directory name
    Argument:
        logdir: the prefix directory
        raw_run_name(str): the base name
    Returns:
        log_path: a non-existent path like logdir/raw_run_name_xxxx
                  where xxxx is an int
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path

        i = i + 1


def plot_confusion_matrix(model, dataloader, classes, device):
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(20,20))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # Wandb 
    wandb.log({"confusion_matrix": wandb.Image(plt)})

