import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.ip_tables import session_1_dict, session_5_dict
from dataset import CustomDataset
from model import Network
from train import train

train_datasets = [CustomDataset("data/session-1/packets.csv", session_1_dict, 0, use_cache=False, split=False)]
train_dataloaders = [DataLoader(x, shuffle=False, batch_size=64, drop_last=True) for x in train_datasets]

test_datasets = [CustomDataset("data/session-5/packets.csv", session_5_dict, 1, use_cache=False, split=False)]
test_dataloaders = [DataLoader(x, shuffle=False, batch_size=64, drop_last=True) for x in test_datasets]

x, ip, port, y = next(iter(train_dataloaders[0]))


def pre_train():
    model = Network()

    epochs = 30
    optim_fn = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optim_fn, 0.0001, epochs=epochs,
                                                    steps_per_epoch=len(train_dataloaders[0]))
    total_loss = []
    avg_losses = []

    for epoch in range(epochs):
        loss_arr = train(model, train_dataloaders[0], optim_fn, loss_fn, scheduler, verbose=True)
        total_loss += loss_arr
        avg_loss = sum(loss_arr) / len(loss_arr)
        print("AVG LOSS: ", avg_loss)
        avg_losses.append(avg_loss)

    print("\nTOTAL LOSS: ", sum(avg_losses) / len(avg_losses), "\n")


def start_training():
    model = Network()

    epochs = 30
    optim_fn = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optim_fn, 0.0001, epochs=epochs,
                                                    steps_per_epoch=len(train_dataloaders[0]))
    for epoch in range(epochs):
        model.train()
        print(f'\n Epochs: {epoch + 1}/{epochs} \n')
        total_loss = []
        for dataloader in train_dataloaders:
            loss_arr = train(model, dataloader, optim_fn, loss_fn, scheduler=scheduler, verbose=False)
            avg_loss = sum(loss_arr) / len(loss_arr)
            print(avg_loss)

            total_loss += loss_arr
        torch.save(model, "trained_models/model_v1.pt")


def validate():
    model = torch.load('trained_models/model_v1.pt')
    model.eval()
    y_pred = torch.Tensor([])
    correct = 0
    for dataloader in test_dataloaders:
        mat = torch.Tensor([[0, 0], [0, 0]])
        hidden = model.init_hidden()
        total = 0
        for idx, (x, ip, port, y) in enumerate(dataloader):
            total += len(x)
            if idx % 100 == 0: print(f'{math.floor(idx * 64 / len(dataloader.dataset) * 100)}%')
            out = model(x.float(), ip, port)
            hidden.detach_()
            out = out.argmax(dim=1)
            y_pred = torch.cat((y_pred, out))
            correct += sum(out == y)
            for a, p in zip(y, out):
                if a == 0 and p == 0:
                    mat[0, 0] += 1
                elif a == 0 and p == 1:
                    mat[0, 1] += 1
                elif a == 1 and p == 0:
                    mat[1, 0] += 1
                else:
                    mat[1, 1] += 1
        mat = mat.long()
        torch.save(y_pred, "./preds.pt")
        print("\t\t\t\tPRED 0 \t PRED 1")
        print("ACTUAL 0 \t ", mat[0, 0].item(), " \t ", mat[0, 1].item())
        print("ACTUAL 1 \t ", mat[1, 0].item(), " \t    ", mat[1, 1].item())
        print(f'\n Accuracy: {correct}/{total}  [{correct / total * 100}] \n')


def interactive_validate():
    print("\n########## INTERACTIVE VALIDATION ########\n")
    model = torch.load('trained_models/model_v1.pt')
    model.eval()

    for dataloader in test_dataloaders:
        total = math.floor(len(dataloader.dataset) / 64)
        jump = None
        hidden = model.init_hidden()
        for batch, (x, ip, port, y) in enumerate(dataloader):
            if jump and batch < jump:
                continue
            else:
                jump = None
            hidden = hidden.detach_()
            out = model(x.float(), ip, port)
            out = out.argmax(dim=1)
            correct = sum(out == y)
            print("ACTUAL: \n", y)
            print("PREDICTED: \n", out)
            print(
                f'{correct}/{len(x)} correct, {correct / len(x) * 100}% [0: {sum(y == 0)}/{sum(out == 0)} 1: {sum(y == 1)}/{sum(out == 1)}]')
            print(f'batch {batch}/{total}')
            i = input("Next?")
            print("\n")

            if not i == "":
                jump = int(i)


# find_lr()
# pre_train()
# start_training()
validate()
# interactive_validate()
