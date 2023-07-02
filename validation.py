import math
import time

import torch


def validate(test_dataloaders, encoder, decoder, csv_writer=None, verbose=False):
    encoder.eval()
    decoder.eval()

    if csv_writer: csv_writer.writerow(['DATASET', 'TIME TAKEN', 'ACCURACY', 'MAT-00', 'MAT-01', 'MAT-10', 'MAT-11'])

    print("\n### VALIDATION ###\n")

    time_elapses = []
    accuracies = []
    final_matrix = torch.Tensor([[0, 0], [0, 0]])

    for dataloader in test_dataloaders:
        print(dataloader.dataset.desc)

        correct = 0
        total = 0
        mat = torch.Tensor([[0, 0], [0, 0]])
        hidden = encoder.init_hidden()
        cell = encoder.init_cell()

        tic = time.perf_counter()
        for idx, (x, ip, port, y) in enumerate(dataloader):
            if idx % 100 == 0: print(
                f'{math.floor(idx * dataloader.dataset.batch_size / len(dataloader.dataset) * 100)}%')

            total += len(x)

            out, (hidden, cell) = encoder(x.float(), ip, port, hidden, cell)
            out, (hidden, cell) = decoder(out, hidden, cell)

            cell.detach_()

            hidden.detach_()

            out = out.argmax(dim=1)
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
        toc = time.perf_counter()

        elapsed_time = math.floor(toc - tic)
        accuracy = round((correct / total * 100).item(), 3)

        accuracies.append(accuracy)
        time_elapses.append(elapsed_time)

        mat = mat.long()
        content = f'\t\t\t\tPRED 0 \t PRED 1' \
                  f'\n' \
                  f'ACTUAL 0 \t {mat[0, 0].item()} \t {mat[0, 1].item()}' \
                  f'\n' \
                  f'ACTUAL 1 \t {mat[1, 0].item()} \t {mat[1, 1].item()}' \
                  f'\n' \
                  f'Accuracy: {correct}/{total} [{accuracy}%]' \
                  f'\n'

        final_matrix[0, 0] += mat[0, 0]
        final_matrix[0, 1] += mat[0, 1]
        final_matrix[1, 0] += mat[1, 0]
        final_matrix[1, 1] += mat[1, 1]

        if verbose: print(content)

        if csv_writer:
            csv_writer.writerow(
                [dataloader.dataset.name, elapsed_time, accuracy, mat[0, 0].item(), mat[0, 1].item(), mat[1, 0].item(),
                 mat[1, 1].item()])

    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_elapsed_time = sum(time_elapses) / len(time_elapses)
    matrix_str = f'\t\t\t\tPRED 0 \t PRED 1' \
                 f'\n' \
                 f'ACUTAL 0 \t {final_matrix[0, 0].item()} \t {final_matrix[0, 1].item()}' \
                 f'\n' \
                 f'ACTUAL 1 \t {final_matrix[1, 0].item()} \t {final_matrix[1, 1].item()}' \
                 f'\n' \
                 f'Final Accuracy: {avg_accuracy}'
    return avg_accuracy, avg_elapsed_time, final_matrix, matrix_str


def interactive_validate(encoder, decoder, test_dataloaders):
    print("\n########## INTERACTIVE VALIDATION ########\n")
    encoder.eval()
    decoder.eval()

    for dataloader in test_dataloaders:
        total = math.floor(len(dataloader.dataset) / 64)
        jump = None
        hidden = encoder.init_hidden()
        for batch, (x, ip, port, y) in enumerate(dataloader):
            if jump and batch < jump:
                continue
            else:
                jump = None
            out, hidden = encoder(x.float(), ip, port, hidden)
            out, hidden = decoder(out, hidden)
            hidden = hidden.detach_()
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
