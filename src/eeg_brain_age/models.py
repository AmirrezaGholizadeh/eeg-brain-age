"""Neural network architectures for EEG brain-age classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedCNN(nn.Module):
    """Shared CNN encoder applied independently to each EEG channel."""

    def __init__(self, output_len=32):
        super(SharedCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            1, 8, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(
            8, 16, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 16 * 3, output_len)
        self.bn3 = nn.BatchNorm1d(output_len)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = F.relu(self.bn3(self.fc(x)))

        return x


class CNNLSTMModel(nn.Module):
    """CNN feature extraction followed by a bidirectional LSTM."""

    def __init__(
        self,
        cnn_output_size=32,
        lstm_hidden_size=32,
        num_classes=9,
        num_channels=21,
    ):
        super(CNNLSTMModel, self).__init__()

        self.shared_cnn = SharedCNN(
            output_len=cnn_output_size
        )

        self.lstm = nn.LSTM(
            input_size=cnn_output_size * num_channels,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(
            lstm_hidden_size * 2,
            num_classes,
        )

    def forward(self, x):
        batch_size, seq_len, num_ch, height, width = x.size()

        outputs = []

        for block_idx in range(seq_len):
            block = x[:, block_idx]
            channel_outs = []

            for channel_idx in range(num_ch):
                channel_input = block[:, channel_idx].unsqueeze(1)
                channel_output = self.shared_cnn(channel_input)
                channel_outs.append(channel_output)

            block_out = torch.cat(channel_outs, dim=1)
            outputs.append(block_out)

        cnn_seq = torch.stack(outputs, dim=1)

        lstm_out, _ = self.lstm(cnn_seq)
        lstm_last = lstm_out[:, -1, :]

        return self.fc(lstm_last)
