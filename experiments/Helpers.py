import math

from train import train
from validation import validate


def run_experiment(train_dataloaders, test_dataloaders, epochs, encoder, decoder, optim_fn, loss_fn, scheduler):
    total_losses = []
    total_elapsed_times = []

    for epoch in range(1, epochs + 1):
        print(f'EPOCH: {epoch} \n')

        loss, elapsed_time = train(encoder, decoder, train_dataloaders, optim_fn, loss_fn, scheduler, verbose=True)

        total_losses.append(loss)
        total_elapsed_times.append(elapsed_time)

    validation_accuracy, validation_time, matrix, matrix_str = validate(test_dataloaders, encoder, decoder)

    total_loss = round(sum(total_losses) / len(total_losses), 3)
    total_elapsed_time = math.floor(sum(total_elapsed_times)) / len(total_elapsed_times)

    return total_loss, total_elapsed_time, validation_accuracy, validation_time, matrix, matrix_str
