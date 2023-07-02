import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.ip_tables import session_5_dict
from dataset import CustomDataset
from model import Encoder, Decoder
from train import train

test_datasets = [CustomDataset("data/session-5/packets.csv", session_5_dict, 4, use_cache=False, split=False)]
test_dataloaders = [DataLoader(x, shuffle=False, batch_size=64, drop_last=True) for x in test_datasets]


def pre_train():
    encoder = Encoder()
    decoder = Decoder()

    epochs = 30
    learning_rate = 0.1
    optim_fn = torch.optim.Adam([
        {'params': encoder.parameters(), 'lr': learning_rate},
        {'params': decoder.parameters(), 'lr': learning_rate}
    ])
    loss_fn = nn.CrossEntropyLoss()
    selected = 1.28E-1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optim_fn, 0.0001, epochs=epochs,
                                                    steps_per_epoch=len(train_dataloaders[0]))
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optim_fn, 0.002, 0.009, cycle_momentum=False)
    total_loss = []
    avg_losses = []

    for epoch in range(epochs):
        loss_arr = train(encoder, decoder, train_dataloaders[0], optim_fn, loss_fn, scheduler, verbose=True)
        total_loss += loss_arr
        avg_loss = sum(loss_arr) / len(loss_arr)
        print("AVG LOSS: ", avg_loss)
        avg_losses.append(avg_loss)

    print("\nTOTAL LOSS: ", sum(avg_losses) / len(avg_losses), "\n")


def start_training():
    encoder = Encoder()
    decoder = Decoder()

    epochs = 30
    learning_rate = 0.0001
    optim_fn = torch.optim.Adam([
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ])
    loss_fn = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optim_fn, 0.0001, epochs=epochs,
                                                    steps_per_epoch=len(train_dataloaders[0]))
    for epoch in range(epochs):
        encoder.train()
        print(f'\n Epochs: {epoch + 1}/{epochs} \n')
        total_loss = []
        for dataloader in train_dataloaders:
            loss_arr = train(encoder, decoder, dataloader, optim_fn, loss_fn, scheduler=scheduler, verbose=False)
            avg_loss = sum(loss_arr) / len(loss_arr)
            print(avg_loss)

            total_loss += loss_arr
            # time.sleep(1)
        # plt.plot(total_loss)
        # plt.show()

        # save model
        torch.save(encoder, "trained_models/encoder.pt")
        torch.save(decoder, "trained_models/decoder.pt")


def validate():
    encoder = torch.load('trained_models/encoder.pt')
    decoder = torch.load('trained_models/decoder.pt')
    encoder.eval()
    decoder.eval()

    y_pred = torch.Tensor([])
    correct = 0
    for dataloader in test_dataloaders:
        # mat = torch.Tensor([[0, 0], [0, 0]])
        hidden = encoder.init_hidden()
        total = 0
        for idx, (x, ip, port, y) in enumerate(dataloader):
            total += len(x)
            if idx % 100 == 0: print(f'{math.floor(idx * 64 / len(dataloader.dataset) * 100)}%')
            out, hidden = encoder(x.float(), ip, port, hidden)
            out, hidden = decoder(out, hidden)
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
    encoder = torch.load('trained_models/encoder.pt')
    decoder = torch.load('trained_models/decoder.pt')
    encoder.eval()
    decoder.eval()

    for dataloader in test_dataloaders:
        total = len(dataloader.dataset)
        jump = None
        hidden = encoder.init_hidden()
        for batch, (x, ip, port, y) in enumerate(dataloader):
            if jump and batch < jump:
                continue
            else:
                jump = None
            out, hidden = encoder(x.float(), ip, port, hidden)
            out, hidden = decoder(out, hidden)
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
