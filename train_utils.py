'''
This module contains methods for training models with different loss functions.
'''

import torch
from torch.nn import functional as F
from torch import nn
if torch.__version__ >= '1.6.0':
    from torch.cuda.amp import autocast
from Losses.loss import cross_entropy, focal_loss, focal_loss_adaptive, adafocal
from Losses.loss import mmce, mmce_weighted
from Losses.loss import brier_score


loss_function_dict = {
    'cross_entropy': cross_entropy,
    'focal_loss': focal_loss,
    'focal_loss_adaptive': focal_loss_adaptive,
    'mmce': mmce,
    'adafocal': adafocal,
    'mmce_weighted': mmce_weighted,
    'brier_score': brier_score
}


def train_single_epoch(epoch,
                       model,
                       train_loader,
                       optimizer,
                       device,
                       loss_function='cross_entropy',
                       gamma=1.0,
                       lamda=1.0,
                       loss_mean=False,
                       scaler=None):
    '''
    Util method for training a model for a single epoch.
    '''
    log_interval = 10
    model.train()
    train_loss = 0
    num_samples = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if scaler is not None: # use mixed precision training
            with autocast():
                logits = model(data)
                if ('mmce' in loss_function):
                    loss = (len(data) * loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device))
                else:
                    loss = loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device)

                if loss_mean:
                    loss = loss / len(data)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) # this is to perform the gradient clipping in the original scale!
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        else:
            logits = model(data)
            if ('mmce' in loss_function):
                loss = (len(data) * loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device))
            else:
                loss = loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device)

            if loss_mean:
                loss = loss / len(data)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            train_loss += loss.item()
            optimizer.step()

        num_samples += len(data)

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / num_samples))
    return train_loss / num_samples



def test_single_epoch(epoch,
                      model,
                      test_val_loader,
                      device,
                      loss_function='cross_entropy',
                      gamma=1.0,
                      lamda=1.0,
                      use_amp=False):
    '''
    Util method for testing a model for a single epoch.
    '''
    model.eval()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_val_loader):
            data = data.to(device)
            labels = labels.to(device)

            if use_amp: # use mixed precision training
                with autocast():
                    logits = model(data)
                    if ('mmce' in loss_function):
                        loss += (len(data) * loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device).item())
                    else:
                        loss += loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device).item()
            else:
                logits = model(data)
                if ('mmce' in loss_function):
                    loss += (len(data) * loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device).item())
                else:
                    loss += loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, device=device).item()
            num_samples += len(data)

    print('======> Test set loss: {:.4f}'.format(
        loss / num_samples))
    return loss / num_samples