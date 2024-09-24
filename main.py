# Standard imports
import logging
import sys
import os
import pathlib
import torch.nn.functional as F
# external imports
import wandb
import yaml
import torch

# import torchinfo.torchinfo as torchinfo
import torch.nn as nn

# locals imports
import models
import Optim_Loss
import dataloader
import utils


def train(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

# start a new wandb run to track this script
    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        run_name = wandb_config.get("run_name", None)  # retrieve run name from config

        # Check if run_name exists in the configuration file
        if run_name is None:
            logging.warning("Run name not specified in the configuration file. A random name will be used.")
            run_name = wandb.util.generate_id()  # generate a random name

        wandb.init(
            # set the wandb project where this run will be logged
            project=wandb_config["project"],
            # set the wandb group
            entity=wandb_config["entity"],
            # set the wandb run name
            name=run_name,
        )
        wandb_log = wandb.log
        logging.info("= Set up the dashboard")

        # Save the configuration in Wandb
        wandb.config.update(config)
    else:
        wandb_log = None


    # creer un datalaoder a parti du fichier config

    # creer un datalaoder a parti du fichier config
    logging.info("= Building the dataloaders")
    (
        train_lod,
        valid_lod,
        inputsize,
        numclasses,
        classes,
        _,
        _,
        _,
    ) = dataloader.get_dataloaders(config, False)
    
    # print(inputsize)
    # print(numclasses, type(numclasses))

    # print(numclasses, type(numclasses))

    # build a moodel
    logging.info("= Building the model")
    modell= models.build_model(config, inputsize, numclasses)
    modell = modell.to(device)

    # build a optimizer, loss
    logging.info("= Building the loss function")
    loss = Optim_Loss.loss_function(config)

    logging.info("= Building the optimizers")
    optim = Optim_Loss.get_optimizer(config, modell.parameters())
    scheduler=Optim_Loss.get_scheduler(optim)
    # définition du checkpoint
    logging.info("= Building the checkpoint")

    logdir = utils.generate_unique_logpath(
        config["logging"]["logdir"], config["model"]["class"]
    )
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    model_checkpoint = utils.ModelCheckpoint(modell, logdir + "/best_model.pt")

    # entrainement du model

    early_stopping_patience = config["training"].get("early_stopping_patience", 8)  # default to 8 if not in config
    epochs_without_improvement = 0

    epochs = config["training"]["nepochs"]
    for i in range(epochs):
        print(i)
        train_loss = utils.train(config, modell, loss, optim,scheduler, device, train_lod)
        valid_loss, accuracy = utils.test(modell, loss, device, valid_lod)

        

        # update le checkpoint and track it for early stopping
        old_min_loss = model_checkpoint.min_loss
        model_checkpoint.update(valid_loss)

        # Check for early stopping condition
        if model_checkpoint.min_loss < old_min_loss:
            epochs_without_improvement = 0  # reset counter
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print("Early stopping due to no improvement after", early_stopping_patience, "epochs.")
            break



        # log metrics to wandb
        wandb.log(
            {"train_loss": train_loss, "valid_loss": valid_loss, "accuracy": accuracy}
        )
    
    utils.plot_confusion_matrix(modell, valid_lod, classes, device)

    # enregistrement du meilleur modèle
    wandb.save(logdir + "/best_model.pt")

    # finish the wandb run
    wandb.finish()

    #utils.plot_confusion_matrix(modell, valid_lod, classes, device)


# a
def test(config):
    pass

    """
    Ajouter dashboard et truc logging 
    
    """


if __name__ == "__main__":

    if len(sys.argv) not in [1, 2]:
        print(f"Usage : {sys.argv[0]} <Config.yaml>")
        sys.exit(-1)

    if len(sys.argv) == 2:
        configpath = sys.argv[1]
    else:
        configpath = "Config.yaml"

    config = yaml.safe_load(open(configpath, "r"))
    train(config)
