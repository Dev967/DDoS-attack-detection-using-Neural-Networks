import math

import torch


def train(encoder, decoder, dataloader, optim_fn, loss_fn, scheduler=None, verbose=False):
    total = len(dataloader.dataset)
    loss_arr = []
    encoder.train()
    decoder.train()

    hidden = torch.zeros(1, 1, 128)
    for idx, (x, ip, port, y) in enumerate(dataloader):
        if x.shape[0] < 64: continue
        out, hidden = encoder(x.float(), ip, port, hidden)
        out, hidden = decoder(out, hidden)
        hidden.detach_()
        loss = loss_fn(out, y)
        optim_fn.zero_grad()
        loss.backward()
        optim_fn.step()
        if scheduler: scheduler.step()

        loss_arr.append(loss.item())

        if verbose and idx % 100 == 0: print(
            f'Loss: {loss.item()}   [{idx * 64}/{total}]({math.floor(idx * 64 / total * 100)}%)')
    return loss_arr
