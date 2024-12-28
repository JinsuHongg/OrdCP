import torch
import torch.nn as nn
import numpy as np


def find_pred(pred, thred: float = 0.5):
    pred = np.where(pred > 0.5, 1, 0)

    output = np.empty((0, 1), int)
    for arr in pred:
        try:
            output = np.append(output, [np.where(arr == 1)[0][-1]])
        except:
            output = np.append(output, [0])
        
        # print(arr, output)

    return output

def train_loop(dataloader, model, loss_fn, optimizer=None, lr_scheduler=None):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)

    pred_arr = np.empty((0, 2), int)
    train_loss = 0
    for batch, (X, y, label) in enumerate(dataloader):
        
        X, y = X.cuda(), y.cuda()
        # Compute prediction and loss
        pred = model(X)
        # _, predictions = torch.max(pred, 1)

        # loss
        # loss = loss_fn(pred, y)#.float()
        loss = loss_fn(pred, y) # log_softmax after NLLloss! torch.nn.functional.log_softmax(pred, dim=1)

        # predictions and true labels
        pred_val = np.reshape( find_pred(pred.cpu().detach().numpy()), (-1, 1))
        label = np.reshape(label.cpu().detach().numpy(), (-1, 1))

        pred_arr = np.append(pred_arr, np.concatenate((pred_val, label), axis=1), axis=0)
      
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        # if batch % 100 == 0:
        #     loss, current = loss.item(), (batch + 1) * len(X)
        #     print(f"loss: {loss:.4f}  [{current:>5d}/{size:>5d}]")

        train_loss += loss
    
    # check loss
    train_loss /= num_batches
    # print(f'Training Avg loss: {train_loss:.4f}')

    return float(train_loss), pred_arr

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(dataloader)
    
    test_loss = 0
    pred_arr = np.empty((0, 2), int)
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y, label in dataloader:
            X, y = X.cuda(), y.cuda()
            pred = model(X)
            _, predictions = torch.max(pred, 1)
            test_loss += loss_fn(pred, y).item() #torch.nn.functional.log_softmax(pred, dim=1)
            
            # predictions and true labels
            pred_val = np.reshape( find_pred(pred.cpu().detach().numpy()), (-1, 1))
            label = np.reshape(label.cpu().detach().numpy(), (-1, 1))

            pred_arr = np.append(pred_arr, np.concatenate((pred_val, label), axis=1), axis=0)

    test_loss /= num_batches

    # print(f"Test loss: {test_loss:.4f}")
    return test_loss, pred_arr

def test_loop_cp(dataloader, model, loss_fn, score_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(dataloader)
    
    test_loss = 0
    pred_arr = np.empty((0,8), float)
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.cuda(), y.cuda()
            pred = model(X)
            _, predictions = torch.max(pred, 1)
            # output = score_fn(pred)
            test_loss += loss_fn(pred, y).item() #torch.nn.functional.log_softmax(pred, dim=1)
            pred_arr = np.append(pred_arr, np.concatenate( (pred.cpu().detach().numpy(), y.cpu().detach().numpy()), axis=1), axis=0)

    test_loss /= num_batches

    # print(f"Test loss: {test_loss:.4f}")
    return test_loss, pred_arr