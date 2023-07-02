import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.Helpers import run_experiment
from model import Encoder, Decoder
from validation import validate


def find_hidden_size(train_dataloaders, test_dataloaders, epochs, optim_fn, lr, loss_fn):
    hidden_sizes = np.arange(128, 896, 128)

    path = 'experiment_outputs/HiddenSize_finder'
    os.makedirs(path, exist_ok=True)

    table_file = open(f'{path}/table.csv', 'w')
    table = csv.writer(table_file)

    table.writerow(
        ['HIDDEN SIZE', 'TRAIN DATA LENGTH', 'TRAIN TIME', 'TOTAL LOSS', 'TEST DATALOADER LENGTH', 'VALIDATION TIME',
         'ACCURACY'])

    for experiment_id, hidden_size in enumerate(hidden_sizes):
        print(f'\n----- EXPERIMENT {experiment_id + 1}/{len(hidden_sizes)} ------\n')
        encoder = Encoder(1, hidden_size, 1, path=path)
        decoder = Decoder(1, hidden_size, 1, path=path)

        optimizer = optim_fn([
            {'params': encoder.parameters()},
            {'params': decoder.parameters()}
        ])

        steps = sum([len(x) for x in train_dataloaders])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps)

        total_loss, training_time, accuracy, validation_time, matrix, matrix_str = run_experiment(train_dataloaders,
                                                                                                  test_dataloaders,
                                                                                                  epochs, encoder,
                                                                                                  decoder, optimizer,
                                                                                                  loss_fn,
                                                                                                  scheduler=scheduler)

        table.writerow(
            [hidden_size, len(train_dataloaders), training_time, total_loss, len(test_dataloaders), validation_time,
             accuracy])
        matrix_file = open(f'{path}/matrix_{hidden_size}.txt', 'w')
        matrix_file.write(matrix_str)
        matrix_file.close()


def find_num_layers(train_dataloaders, test_dataloaders, epochs, optim_fn, lr, loss_fn):
    num_layers = np.arange(1, 12, 2)

    path = 'experiment_outputs/NumLayers_finder'
    os.makedirs(path, exist_ok=True)

    table_file = open(f'{path}/table.csv', 'w')
    table = csv.writer(table_file)

    table.writerow(
        ['NUM LAYERS', 'TRAIN DATA LENGTH', 'TRAIN TIME', 'TOTAL LOSS', 'TEST DATALOADER LENGTH', 'VALIDATION TIME',
         'ACCURACY'])

    for experiment_id, num_layer in enumerate(num_layers):
        print(f'\n----- EXPERIMENT {experiment_id + 1}/{len(num_layers)} ------\n')
        encoder = Encoder(num_layer, 128, 1, path=path)
        decoder = Decoder(num_layer, 128, 1, path=path)

        optimizer = optim_fn([
            {'params': encoder.parameters()},
            {'params': decoder.parameters()}
        ])

        steps = sum([len(x) for x in train_dataloaders])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps)

        total_loss, training_time, accuracy, validation_time, matrix, matrix_str = run_experiment(train_dataloaders,
                                                                                                  test_dataloaders,
                                                                                                  epochs, encoder,
                                                                                                  decoder, optimizer,
                                                                                                  loss_fn,
                                                                                                  scheduler=scheduler)

        table.writerow(
            [num_layer, len(train_dataloaders), training_time, total_loss, len(test_dataloaders), validation_time,
             accuracy])
        matrix_file = open(f'{path}/matrix_{num_layer}.txt', 'w')
        matrix_file.write(matrix_str)
        matrix_file.close()


def find_batch_size(train_datasets, test_datasets, epochs, optim_fn, lr, loss_fn):
    batch_sizes = [64, 128, 192, 256, 320, 384, 448, 512]

    path = 'experiment_outputs/Batch_size_finder'
    os.makedirs(path, exist_ok=True)

    table_file = open(f'{path}/table.csv', 'w')
    table = csv.writer(table_file)

    table.writerow(
        ['BATCH SIZE', 'TRAIN DATA LENGTH', 'TRAIN TIME', 'TOTAL LOSS', 'TEST DATALOADER LENGTH', 'VALIDATION TIME',
         'ACCURACY'])

    for experiment_id, batch_size in enumerate(batch_sizes):
        train_dataloaders = [DataLoader(x, shuffle=False, batch_size=batch_size, drop_last=True) for x in
                             train_datasets]
        test_dataloaders = [DataLoader(x, shuffle=False, batch_size=batch_size, drop_last=True) for x in test_datasets]

        print(f'\n----- EXPERIMENT {experiment_id + 1}/{len(batch_sizes)} ------\n')
        encoder = Encoder(1, 128, 1, path=path, batch_size=batch_size)
        decoder = Decoder(1, 128, 1, path=path, batch_size=batch_size)

        optimizer = optim_fn([
            {'params': encoder.parameters()},
            {'params': decoder.parameters()}
        ])

        steps = sum([len(x) for x in train_dataloaders])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps)

        total_loss, training_time, accuracy, validation_time, matrix, matrix_str = run_experiment(train_dataloaders,
                                                                                                  test_dataloaders,
                                                                                                  epochs, encoder,
                                                                                                  decoder, optimizer,
                                                                                                  loss_fn,
                                                                                                  scheduler=scheduler)

        table.writerow(
            [batch_size, len(train_dataloaders), training_time, total_loss, len(test_dataloaders), validation_time,
             accuracy])
        matrix_file = open(f'{path}/matrix_{batch_size}.txt', 'w')
        matrix_file.write(matrix_str)
        matrix_file.close()


def find_activation(train_dataloaders, test_dataloaders, epochs, optim_fn, lr, loss_fn):
    activations = [nn.ELU, nn.ReLU, nn.LeakyReLU, nn.MultiheadAttention, nn.PReLU, nn.RReLU, nn.GELU, nn.SiLU,
                   nn.Tanh]

    path = 'experiment_outputs/Activation_finder'
    os.makedirs(path, exist_ok=True)

    table_file = open(f'{path}/table.csv', 'w')
    table = csv.writer(table_file)

    table.writerow(
        ['ACTIVATION', 'TRAIN DATA LENGTH', 'TRAIN TIME', 'TOTAL LOSS', 'TEST DATALOADER LENGTH', 'VALIDATION TIME',
         'ACCURACY'])

    for experiment_id, activation in enumerate(activations):
        print(f'\n----- EXPERIMENT {experiment_id + 1}/{len(activations)} ------\n')
        encoder = Encoder(1, 128, 1, path=path, activation_fn=activation())
        decoder = Decoder(1, 128, 1, path=path, activation_fn=activation())

        optimizer = optim_fn([
            {'params': encoder.parameters()},
            {'params': decoder.parameters()}
        ])

        steps = sum([len(x) for x in train_dataloaders])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps)

        total_loss, training_time, accuracy, validation_time, matrix, matrix_str = run_experiment(train_dataloaders,
                                                                                                  test_dataloaders,
                                                                                                  epochs, encoder,
                                                                                                  decoder, optimizer,
                                                                                                  loss_fn,
                                                                                                  scheduler=scheduler)

        table.writerow(
            [activation.__name__, len(train_dataloaders), training_time, total_loss, len(test_dataloaders),
             validation_time,
             accuracy])
        matrix_file = open(f'{path}/matrix_{activation.__name__}.txt', 'w')
        matrix_file.write(matrix_str)
        matrix_file.close()


def find_dropout(train_dataloaders, test_dataloaders, epochs, optim_fn, lr, loss_fn):
    dropouts = np.arange(0.1, 1, 0.1)

    path = 'experiment_outputs/Dropout_finder'
    os.makedirs(path, exist_ok=True)

    table_file = open(f'{path}/table.csv', 'w')
    table = csv.writer(table_file)

    table.writerow(
        ['DROPOUT', 'TRAIN DATA LENGTH', 'TRAIN TIME', 'TOTAL LOSS', 'TRAIN DATA ACCURACY', 'TEST DATALOADER LENGTH',
         'VALIDATION TIME',
         'ACCURACY'])

    for experiment_id, dropout in enumerate(dropouts):
        print(f'\n----- EXPERIMENT {experiment_id + 1}/{len(dropouts)} ------\n')
        encoder = Encoder(2, 128, 1, path=path, dropout=dropout)
        decoder = Decoder(2, 128, 1, path=path, dropout=dropout)

        optimizer = optim_fn([
            {'params': encoder.parameters()},
            {'params': decoder.parameters()}
        ])

        steps = sum([len(x) for x in train_dataloaders])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps)

        total_loss, training_time, accuracy, validation_time, matrix, matrix_str = run_experiment(train_dataloaders,
                                                                                                  test_dataloaders,
                                                                                                  epochs, encoder,
                                                                                                  decoder, optimizer,
                                                                                                  loss_fn,
                                                                                                  scheduler=scheduler)

        train_data_accuracy, _, _, _ = validate(train_dataloaders, encoder, decoder)

        table.writerow(
            [dropout, len(train_dataloaders), training_time, total_loss, train_data_accuracy, len(test_dataloaders),
             validation_time,
             accuracy])
        matrix_file = open(f'{path}/matrix_{dropout}.txt', 'w')
        matrix_file.write(matrix_str)
        matrix_file.close()
