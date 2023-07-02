import copy
import csv

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from train import train


def learning_rate_finder(train_dataloader, enc, dec):
    learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    optimizers = [torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW, torch.optim.NAdam]
    path = 'experiment_outputs/LR_finder'
    table_file = open(f'{path}/table.csv', 'w')
    table = csv.writer(table_file)
    table.writerow(['SCHEDULER', 'OPTIMIZER', 'LEARNING RATE', 'AVG LOSS'])

    for scheduler in [None, torch.optim.lr_scheduler]:
        for optimizer in optimizers:
            avg_losses = []
            for learning_rate in learning_rates:
                encoder = copy.deepcopy(enc)
                decoder = copy.deepcopy(dec)

                optim = optimizer([
                    {'params': encoder.parameters(), 'lr': learning_rate},
                    {'params': decoder.parameters(), 'lr': learning_rate}
                ])
                optim_name = optimizer.__name__

                loss_fn = nn.CrossEntropyLoss()

                scheduler_name = None
                if scheduler:
                    lr_scheduler = scheduler(optim, learning_rate, epochs=1, steps_per_epoch=len(train_dataloader))
                    scheduler_name = scheduler.__name__
                else:
                    lr_scheduler = None

                loss_arr = train(encoder, decoder, train_dataloader, optim, loss_fn, scheduler=lr_scheduler)

                plt.clf()
                plt.plot(loss_arr)
                plt.savefig(f'{path}/{scheduler_name}_{optim_name}_{str(learning_rate).replace(".", "_")}.png')

                avg_loss = sum(loss_arr) / len(loss_arr)
                avg_losses.append(avg_loss)
                table.writerow([scheduler_name, optim_name, str(learning_rate), str(avg_loss)])
            plt.plot(avg_losses)
            plt.savefig(f'{path}/{scheduler_name}_{optim_name}_total.png')
