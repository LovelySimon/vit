import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple


# 训练步骤
def train_step(model:torch.nn.Module,dataloader:torch.utils.data.Dataloader,loss_fn:torch.nn.Module,optimizer:torch.optim.Optimizer,device:torch.device):
    """
    :param model:训练模型
    :param dataloader:
    :param loss_fn: loss函数
    :param optimizer: 优化器
    :param device: 训练模型的设备：cpu or gpu
    :return: (train_loss, train_accuracy)
    """
    model.train(True)
    train_loss = 0.0
    train_accuracy = 0.0
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class=torch.argmax(torch.softmax(y_pred,dim=1), dim=1)
        train_accuracy+=(y_pred_class==y).sum().item()/len(y_pred)
    # 得到平均的loss和准确度
    train_loss=train_loss/len(dataloader)
    train_accuracy=train_accuracy/len(dataloader)
    return train_loss, train_accuracy

# 验证步骤
def val_step(model:torch.nn.Module,dataloader:torch.utils.data.Dataloader,loss_fn:torch.nn.Module,optimizer:torch.optim.Optimizer,device:torch.device):
    """
    验证模型
    :param model:z
    :param dataloader:
    :param loss_fn:
    :param optimizer:
    :param device:
    :return: (val_loss, val_accuracy)
    """
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for batch, (X,y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logists = model(X)
            loss = loss_fn(test_pred_logists, y)
            val_loss+=loss.item()
            val_predict_labels=test_pred_logists.argmax(dim=1)
            val_accuracy += ((val_predict_labels == y).sum().item()/len(val_predict_labels))
    val_loss=val_loss/len(dataloader)
    val_accuracy=val_accuracy/len(dataloader)
    return val_loss, val_accuracy

def train(model:torch.nn.Module, train_loader:torch.utils.data.DataLoader(),val_loader:torch.utils.data.DataLoader(),optimizer:torch.optim.Optimizer, loss_fn:torch.nn.Module,epochs:int,device:torch.device):
    """
    整合
    :param model:
    :param train_loader:
    :param val_loader:
    :param optimizer:
    :param epochs:
    :param device:
    :return:A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]}
    For example if training for epochs=2:
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]}
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss":[],
        "val_acc":[],
    }
    model.to(device)
    # 使用 tqdm 库来显示一个进度条，通常用于迭代训练深度学习模型时的每个epoch。
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,dataloader=train_loader,loss_fn=loss_fn, optimizer=optimizer,device=device)
        val_loss, val_acc = val_step(model=model,dataloader=train_loader,loss_fn=loss_fn, optimizer=optimizer,device=device)
        print(f"Epoch {epoch+1}/{epochs}|"
              f"Train Loss: {train_loss:.4f}|"
              f"Train Acc: {train_acc:.4f}|"
              f"Val Loss: {val_loss:.4f}|"
              f"Val Acc: {val_acc:.4f}")
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

    return results
