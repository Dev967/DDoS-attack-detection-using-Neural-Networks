import math


def train(model, dataloader, optim_fn, loss_fn, scheduler=None, verbose=False):
    total = len(dataloader.dataset)
    loss_arr = []
    model.train()

    for idx, (x, ip, port, y) in enumerate(dataloader):
        if x.shape[0] < 64: continue
        out = model(x.float(), ip, port)
        loss = loss_fn(out, y)
        optim_fn.zero_grad()
        loss.backward()
        optim_fn.step()
        if scheduler: scheduler.step()

        loss_arr.append(loss.item())

        if idx % 100 == 0 and verbose: print(
            f'Loss: {loss.item()}   [{idx * 64}/{total}]({math.floor(idx * 64 / total * 100)}%)')
    return loss_arr
