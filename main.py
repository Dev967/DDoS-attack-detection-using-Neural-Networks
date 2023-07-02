import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.ip_tables import session_1_dict, session_3_dict, session_4_dict, session_5_dict
from dataset import CustomDataset
from model import Encoder, Decoder
from train import train
from validation import validate

train_dataset_attrib = [
    {
        "path": "data/session-1/packets.csv",
        "index": 0,
        "ip_table": session_1_dict,
        "use_cache": True,
        "desc": "Session - 1"
    },
    {
        "path": "data/session-3/packets.csv",
        "index": 2,
        "ip_table": session_3_dict,
        "use_cache": True,
        "desc": "Session - 3"
    },
    {
        "path": "data/session-4/packets.untrimmed.csv",
        "index": 3,
        "ip_table": session_4_dict,
        "use_cache": True,
        "desc": "Session - 4"
    }
]

train_datasets = [
    CustomDataset(x["path"], x["ip_table"], x["index"], use_cache=x["use_cache"], desc=x["desc"], batch_size=64) for x
    in train_dataset_attrib]
train_dataloaders = [DataLoader(x, shuffle=False, batch_size=64, drop_last=True) for x in train_datasets]

test_datasets = [
    CustomDataset("data/session-5/packets.csv", session_5_dict, 4, use_cache=True, desc="Session - 5", batch_size=64)]
test_dataloaders = [DataLoader(x, shuffle=False, batch_size=64, drop_last=True) for x in test_datasets]

epochs = 15
encoder = Encoder(1, 128, 1, activation_fn=nn.LeakyReLU(), batch_size=64, dropout=0.3, name="final_encoder",
                  path="reports/trained_with_changing_hidden")
decoder = Decoder(1, 128, 1, activation_fn=nn.LeakyReLU(), other_activation_fn=nn.Softmax(dim=1), batch_size=64,
                  dropout=0.3, name="final_decoder", path="reports/trained_with_changing_hidden")
encoder.load()
decoder.load()
optimizer = torch.optim.NAdam([
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
])
steps = sum([len(x) for x in train_dataloaders])
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, epochs=epochs, steps_per_epoch=steps)
loss_fn = nn.CrossEntropyLoss()

train_log = open("reports/trained_without_changing_hidden/train_logs.csv", "w")
train_log_writer = csv.writer(train_log)

validation_log = open('reports/trained_without_changing_hidden/validation_logs.csv', 'w')
validation_log_writer = csv.writer(validation_log)

train_log_writer.writerow(["EPOCH", "DATASET", "ELAPSED TIME", "AVG LOSS"])
for epoch in range(1, epochs + 1):
    print(f'EPOCH: {epoch} \n')

    loss, elapsed_time = train(encoder, decoder, train_dataloaders, optimizer, loss_fn, scheduler, verbose=True,
                               csv_writer=train_log_writer)
    encoder.save()
    decoder.save()

validation_accuracy, validation_time, matrix, matrix_str = validate(test_dataloaders, encoder, decoder)
print(validation_accuracy, matrix_str)
