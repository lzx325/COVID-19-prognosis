import random
import os
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.autograd
import numpy as np
import collections
import pickle as pkl
import sklearn.model_selection

def weighted_l2_loss(pred,gt,weight=None,reduction=True):
    assert pred.shape==gt.shape
    device=pred.device
    if  weight  is None:
        weight=torch.ones((pred.shape[1],)).to(device)
    else:
        assert len(weight)==pred.shape[1]
    if reduction:
        mse=torch.mean(((pred-gt)**2).mean(dim=0)*weight)
    else:
        mse=((pred-gt)**2).mean(dim=0)*weight
    return mse

def fit(mynet,X,Y,params,weight=None,verbose=1):
    n_epoch,lr,beta,batch_size=params["n_epoch"],params["lr"],params["beta"],params["batch_size"]
    device=next(mynet.parameters()).device

    mynet.train()

    X_tensor=torch.FloatTensor(X)
    Y_tensor=torch.FloatTensor(Y)
    dataset=torch.utils.data.TensorDataset(X_tensor,Y_tensor)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=batch_size)

    loss_fn=torch.nn.MSELoss()
    optimizer = optim.Adam(mynet.parameters(), lr=lr)

    if weight is not None:
        weight=torch.FloatTensor(weight).to(device)
    for epoch in range(n_epoch):
        for i,(X_batch,Y_batch) in enumerate(dataloader):
            X_batch=X_batch.to(device)
            Y_batch=Y_batch.to(device)
            Y_pred=mynet(X_batch)
            loss_regression=weighted_l2_loss(Y_pred,Y_batch,weight)
            loss_regularization=mynet.l2_regularization()
            loss=loss_regression+beta*loss_regularization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose>0 and epoch%10==0:
                print("Epoch: %d Step: [%d/%d] loss=%.4f"%(epoch,i+1,len(dataloader),loss.item()))

def score(mynet,X,Y,batch_size=128):
    assert len(X)==len(Y)
    n_features=X.shape[1]
    n_targets=Y.shape[1]
    mynet.eval()
    
    X_tensor=torch.FloatTensor(X)
    Y_tensor=torch.FloatTensor(Y)
    dataset=torch.utils.data.TensorDataset(X_tensor,Y_tensor)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=batch_size)

    device=next(mynet.parameters()).device
    scores=dict()
    scores["MSE"]=np.zeros((n_targets,))
    scores["feature_grad"]=np.zeros((n_targets,n_features))
    scores["feature_grad_abs"]=np.zeros((n_targets,n_features))
    for _,(X_batch,Y_batch) in enumerate(dataloader):
        X_batch=X_batch.to(device)
        X_batch.requires_grad_()
        Y_batch=Y_batch.to(device)
        Y_pred=mynet(X_batch)
        mse=weighted_l2_loss(Y_pred,Y_batch,reduction=False)
        Y_pred_sum=Y_pred.sum(0)
        scores["MSE"]+=(mse.detach().cpu().numpy())*len(X_batch)/len(X)
        for j in range(n_targets):
            X_batch_grad=torch.autograd.grad(Y_pred_sum[j],X_batch,retain_graph=True)[0]
            X_batch_grad_np=X_batch_grad.cpu().numpy()
            scores["feature_grad"][j]+=X_batch_grad_np.sum(axis=0)*(1/len(X))
            scores["feature_grad_abs"][j]+=np.abs(X_batch_grad_np).sum(axis=0)*(1/len(X))
    
    mynet.train()
    return scores






