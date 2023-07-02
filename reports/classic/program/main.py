import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from dataset import CustomDataset

test_dataset = CustomDataset('data/session-5/packets.csv')
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)

model = torch.load("trained_models/model_v1.pt")

correct = 0
total = 0
mat = torch.Tensor([[0, 0], [0, 0]])
hidden = model.init_hidden()
y_act = torch.Tensor([])
y_pred = torch.Tensor([])

for idx, (x, y) in enumerate(test_dataloader):
    total += len(x)
    y_act = torch.cat((y_act, y))
    out = model(x.float())

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

accuracy = round((correct / total * 100).item(), 3)

mat = mat.long()
content = f'\t\t\t\tPRED 0 \t PRED 1' \
          f'\n' \
          f'ACTUAL 0 \t {mat[0, 0].item()} \t {mat[0, 1].item()}' \
          f'\n' \
          f'ACTUAL 1 \t {mat[1, 0].item()} \t {mat[1, 1].item()}' \
          f'\n' \
          f'Accuracy: {correct}/{total} [{accuracy}%]' \
          f'\n'

print(accuracy, content)
print(roc_auc_score(y_act.numpy(), y_pred.numpy()))

torch.save(y_pred, "./preds.pt")

fpr, tpr, _ = roc_curve(y_act.numpy(), y_pred.numpy())
auc = roc_auc_score(y_act.numpy(), y_pred.numpy())
plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
plt.legend(loc=4)
plt.show()

plt.show()
