import os
import numpy as np
import random
import datetime
import torch
__all__ = ['save_state_dict', 'load_state_dict', 'load_state_dict_test']


def save_state_dict(model, optimizer, lr_scheduler, train_loss, val_loss, checkpoint_dir, model_architecture, epoch_idx):
    """Save the state dict of the current training to disk

    Parameters
    ----------
    model : torch model
        trained model to save
    optimizer : torch.optim object
    lr_scheduler: torch.optim.lr_scheduler object
    checkpoint_dir : str or os.path object
        directory into which to save the model
    model_architecture : str
        type of architecture
    epoch : int
        epoch index

    Returns
    -------
    str or os.path object
        path to the saved model

    """
    state = dict(
                 model=model.state_dict(),
                 optimizer=optimizer.state_dict(),
                 lr_scheduler=lr_scheduler.state_dict(),
                 epoch=epoch_idx,
                 train_loss=train_loss,
                 val_loss=val_loss,
                 )
    time_stamp = datetime.datetime.now().strftime("epoch={:d}_%m-%d-%Y_%H:%M".format(epoch_idx))
    model_fname = '{:s}_{:s}.mdl'.format(model_architecture, time_stamp)
    model_path = os.path.join(checkpoint_dir, model_fname)
    torch.save(state, model_path)
    return model_path

def load_state_dict(checkpoint_path, model, optimizer, n_epochs, device, lr_scheduler=None):
    """Load the state dict of the past training

    Parameters
    ----------
    checkpoint_path : str or os.path object
        path of the state dict to load
    model : torch model
        trained model to save
    optimizer : torch.optim object
    lr_scheduler: torch.optim.lr_scheduler object
    n_epochs : int
        total number of epochs to train
    device : torch.device object
        device on which to load the model

    Returns
    -------
    str or os.path object
        path to the saved model

    """
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['model'])
    model.to(device)
    optimizer.load_state_dict(state['optimizer'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(state['lr_scheduler'])
    epoch = state['epoch']
    train_loss = state['train_loss']
    val_loss = state['val_loss']
    print("Loaded weights at {:s}".format(checkpoint_path))
    print("Epoch [{}/{}]: TRAIN Loss: {:.4f}".format(epoch+1, n_epochs, train_loss))
    print("Epoch [{}/{}]: VALID Loss: {:.4f}".format(epoch+1, n_epochs, val_loss))
    return epoch, model, optimizer, train_loss, val_loss

def load_state_dict_test(checkpoint_path, model, n_epochs, device):
    """Load the state dict of the past training

    Parameters
    ----------
    checkpoint_path : str or os.path object
        path of the state dict to load
    model : torch model
        trained model to save
    optimizer : torch.optim object
    lr_scheduler: torch.optim.lr_scheduler object
    n_epochs : int
        total number of epochs to train
    device : torch.device object
        device on which to load the model

    Returns
    -------
    str or os.path object
        path to the saved model

    """
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['model'])
    model.to(device)
    epoch = state['epoch']
    train_loss = state['train_loss']
    val_loss = state['val_loss']
    print("Loaded weights at {:s}".format(checkpoint_path))
    print("Epoch [{}/{}]: TRAIN Loss: {:.4f}".format(epoch+1, n_epochs, train_loss))
    print("Epoch [{}/{}]: VALID Loss: {:.4f}".format(epoch+1, n_epochs, val_loss))
    return model, epoch