import math
import time


def train(encoder, decoder, dataloaders, optim_fn, loss_fn, scheduler=None, verbose=False, epoch=None, csv_writer=None):
    encoder.train()
    decoder.train()

    avg_losses = []
    time_elapses = []

    count = 0
    for dataloader_idx, dataloader in enumerate(dataloaders):
        hidden = encoder.init_hidden()
        cell = encoder.init_cell()

        count += 1
        dataloader_total = len(dataloader.dataset)
        print(
            f'Training on: [{dataloader_idx + 1}/{len(dataloaders)}]) {dataloader.dataset.desc}')

        tic = time.perf_counter()
        loss_arr = []
        for idx, (x, ip, port, y) in enumerate(dataloader):
            if x.shape[0] < 64: continue
            # RNN/GRU
            # encoder_outputs, hidden = encoder(x.float(), ip, port, hidden)
            # output, hidden = decoder(encoder_outputs, hidden)
            # hidden.detach_()
            # LSTM
            out, (hidden, cell) = encoder(x.float(), ip, port, hidden, cell)
            output, (hidden, cell) = decoder(out, hidden, cell)
            cell.detach_()
            hidden.detach_()

            loss = loss_fn(output, y)
            optim_fn.zero_grad()
            loss.backward()
            optim_fn.step()
            if scheduler: scheduler.step()

            loss_arr.append(loss.item())

            if verbose and idx % 100 == 0: print(
                f'Loss: {loss.item()}   [{idx * 64}/{dataloader_total}]({math.floor(idx * 64 / dataloader_total * 100)}%)')

        toc = time.perf_counter()

        elapsed_time = math.floor(toc - tic)
        avg_loss = round(sum(loss_arr) / len(loss_arr), 3)

        avg_losses.append(avg_loss)
        time_elapses.append(elapsed_time)

        print(f'\nAverage Loss: {avg_loss}'
              f'\n'
              f'Time taken: {elapsed_time}s'
              f'\n')

        if csv_writer:
            csv_writer.writerow([str(epoch), str(dataloader.dataset.name), str(elapsed_time), str(avg_loss)])

    total_avg_loss = sum(avg_losses) / len(avg_losses)
    total_avg_time = sum(time_elapses) / len(time_elapses)
    return round(total_avg_loss, 3), total_avg_time
