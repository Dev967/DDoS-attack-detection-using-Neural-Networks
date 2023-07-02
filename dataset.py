import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, csv_path, lookup, idx, use_cache=False, desc=None, batch_size=64):
        self.IP_SRC_COLS = ["SRC_IP_1", "SRC_IP_2", "SRC_IP_3", "SRC_IP_4"]
        self.IP_DST_COLS = ["DST_IP_1", "DST_IP_2", "DST_IP_3", "DST_IP_4"]
        self.batch_size = 64

        self.name = desc

        self.train = {}

        if use_cache:
            self.train["x_frame"] = pd.read_csv(f'cache/train_x_frame_{idx}.csv')
            self.train["y_frame"] = pd.read_csv(f'cache/train_y_frame_{idx}.csv')
        else:
            self.dataframe = pd.read_csv(csv_path,
                                         dtype={
                                             "TCP_MSS": 'Int64',
                                             "TCP_NOP": 'Int64',
                                             "TCP_WSCALE": 'Int64',
                                             "TYPE": 'Int64',
                                         }).fillna(0)

            src_cols = []
            dst_cols = []

            for row in self.dataframe.iterrows():
                row = row[1]
                src_cols.append(lookup[row.SRC])
                dst_cols.append(lookup[row.DST])

            self.dataframe[self.IP_SRC_COLS] = src_cols
            self.dataframe[self.IP_DST_COLS] = dst_cols
            del src_cols
            del dst_cols

            # put in or remove columns here
            self.dataframe = self.dataframe.drop(["SRC", "DST", "TIMESTAMP", "TCP_NOP", "TCP_SEQ", "TCP_ACK"], axis=1)

            self.dataframe = self.dataframe.astype('int64')

            self.train["x_frame"] = self.dataframe.drop(['TYPE'], axis=1)
            self.train["y_frame"] = self.dataframe["TYPE"].copy()

            del self.dataframe

            # for caching
            self.train["x_frame"].to_csv(f'cache/train_x_frame_{idx}.csv', index=False)
            self.train["y_frame"].to_csv(f'cache/train_y_frame_{idx}.csv', index=False)

        self.y_frame = self.train["y_frame"]
        self.x_frame = self.train["x_frame"]

        self.y_frame = torch.from_numpy(self.y_frame.to_numpy().reshape(1, -1)).squeeze()
        self.ip_frame = torch.from_numpy(self.x_frame[self.IP_SRC_COLS + self.IP_DST_COLS].copy().to_numpy())
        self.port_frame = torch.from_numpy(self.x_frame[["TCP_SPORT", "TCP_DPORT"]].copy().to_numpy())
        self.x_frame = self.x_frame.drop(self.IP_SRC_COLS + self.IP_DST_COLS + ["TCP_SPORT", "TCP_DPORT"], axis=1)
        self.x_frame = torch.from_numpy(self.x_frame.to_numpy()).float()

        self.desc = f'{desc}\n' \
                    f'Total: {len(self.x_frame)}\n' \
                    f'0 = {sum(self.y_frame == 0)}\n' \
                    f'1 = {sum(self.y_frame == 1)}\n'

    def __len__(self):
        return len(self.x_frame)

    def __getitem__(self, idx):
        return self.x_frame[idx], self.ip_frame[idx], self.port_frame[idx], self.y_frame[idx]
