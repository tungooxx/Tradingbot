import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import numpy as np
class DNN_Net(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=32, num_layers=2, output_size=4, dropout=0.2,conv_out = 32):
        super().__init__()

        self.hidden_layer_size = hidden_layer_size
        self.lstm_input = conv_out
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=conv_out, kernel_size=3)
        self.linear_1 = nn.Linear(self.lstm_input, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_layer_size * num_layers, output_size * num_layers)
        self.dense2 = nn.Linear(output_size * num_layers, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        x = self.linear_1(x)
        x = self.relu(x)

        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        x_dense = self.dense(x)
        x_dense = self.dropout(x_dense)
        predictions = self.dense2(x_dense)

        return predictions


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        x = np.expand_dims(x, 2)
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])