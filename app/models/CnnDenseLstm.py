import torch
import torch.nn as nn

class CnnDenseLstm(nn.Module):
    def __init__(self, input_size=3, cnn_channels=[32, 64], cnn_kernel_sizes=[3, 3],
                 lstm_hidden_size=128, lstm_layers=2, output_size=1, dropout=0.3):
        super().__init__()
        
        # CNN 부분
        cnn_layers = []
        in_channels = input_size
        for out_channels, k in zip(cnn_channels, cnn_kernel_sizes):
            cnn_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k//2))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        self.cnn = nn.Sequential(*cnn_layers)
        
        # LSTM 부분
        self.lstm = nn.LSTM(input_size=cnn_channels[-1],
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True,
                            dropout=dropout)
        
        # Dense 분류기
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, feature) → CNN 입력용으로 (batch, feature, seq_len)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)  # (batch, cnn_channels[-1], seq_len)
        x = x.permute(0, 2, 1)  # LSTM 입력용으로 (batch, seq_len, feature)
        
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 마지막 타임스텝
        out = self.fc(out)
        return out
