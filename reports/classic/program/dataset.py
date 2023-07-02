import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, csv_path, use_cache=False, test=False):

        if use_cache:
            self.train["x_frame"] = pd.read_csv("cache/train_x_frame.csv")
            self.train["y_frame"] = pd.read_csv("cache/train_y_frame.csv")
            self.test["x_frame"] = pd.read_csv("cache/test_x_frame.csv")
            self.test["y_frame"] = pd.read_csv("cache/test_y_frame.csv")
        else:
            self.dataframe = pd.read_csv(csv_path,
                                         dtype={
                                             "TCP_MSS": 'Int64',
                                             "TCP_NOP": 'Int64',
                                             "TCP_WSCALE": 'Int64',
                                             "TYPE": 'Int64',
                                         }).fillna(0)

            self.dataframe = self.dataframe.drop(
                ["SRC", "DST", "TCP_SPORT", "TCP_DPORT", "TIMESTAMP", "TCP_NOP", "TCP_SEQ", "TCP_ACK"], axis=1)

            self.dataframe = self.dataframe.astype('int64')

        self.y_frame = self.dataframe["TYPE"].copy()
        self.x_frame = self.dataframe.drop(["TYPE"], axis=1).copy()

        self.y_frame = torch.from_numpy(self.y_frame.to_numpy().reshape(1, -1)).squeeze()
        self.x_frame = torch.from_numpy(self.x_frame.to_numpy()).unsqueeze(1).float()

    def __len__(self):
        return len(self.x_frame)

    def __getitem__(self, idx):
        return self.x_frame[idx], self.y_frame[idx]
