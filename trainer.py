from torchvision import models
from torch import nn
import torch.optim as optim
import statics as st
import numpy as np
import torch
import optuna
from torch.utils.data import DataLoader


resnet = models.resnet18(pretrained=True)
#for parameter in resnet.parameters():
    #parameter.requires_grad = False

resnet.fc = nn.Linear(512, 2)
resnet_criterion = nn.CrossEntropyLoss()
resnet_optimizer = optim.SGD([parameters for parameters in resnet.parameters() if parameters.requires_grad], lr=0.00016583879478621638, momentum=0.9374218810592992, weight_decay=0.0006603822752434714)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = resnet.to(device)

train_loader = st.train_loader
test_loader = st.test_loader


def train(epochs, model, criterion, optimizer, train_loader, test_loader, device):
    print(device)
    losses = []
    accuracies = []
    test_len = st.test_dataset.__len__()

    for epoch in range(epochs):
        model.train()
        temp_loss = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            temp_loss += loss.item()
        temp_loss = np.mean(temp_loss)
        print(f"Epoch {epoch} Loss: ", temp_loss)
        losses.append(temp_loss)

        model.eval()
        temp_accuracy = 0
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            _, yhat = torch.max(pred.data, 1)
            temp_accuracy += (y == yhat).sum().item()
        temp_accuracy = temp_accuracy/test_len
        print(f"Epoch {epoch} Accuracy: ", temp_accuracy)
        accuracies.append(temp_accuracy)

    return losses, accuracies

def save(model):
    torch.save(model.state_dict(), "fireDetectorAI.pt")

def optunaTrain(epochs, model, criterion, optimizer, train_loader, test_loader, device):
    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            _, yhat = torch.max(pred.data, 1)
            total += y.size(0)
            correct += (yhat == y).sum().item()
    accuracy = correct / total
    return accuracy

def objective(trial):
    batch_size = trial.suggest_int('batch_size', 16, 64)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    momentum = trial.suggest_uniform('momentum', 0.8, 0.99)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-1)

    train_loader = DataLoader(st.train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(st.test_dataset, batch_size=batch_size, shuffle=False)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    accuracy = optunaTrain(10, model, criterion, optimizer, train_loader, test_loader, device)
    print("Accuracy: ", accuracy)
    return accuracy

def findBestHyperParams(n_trials):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print("Best HyperParams: ", study.best_params)
    print("Best Accuracy: ", study.best_value)
    with  open("optimalHyperParams.txt", "x") as f:
        f.write("Best HyperParams: ", study.best_params, "\n\n", "Best Accuracy: ", study.best_value)

train(10, resnet, resnet_criterion, resnet_optimizer, train_loader, test_loader, device)
save(resnet)