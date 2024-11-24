import torch
import torch.nn as nn
import torch.nn.functional as F

class OCRModel(nn.Module):
    def __init__(self, vocab_size: int, input_channels:int = 3, hidden_size:int = 256, num_layers:int = 2, dropout:float = 0.2, sequence_length: int = 64):
        """
        Initializes the OCR Model with adjustable sequence length.
        
        Args:
            vocab_size (int): Size of the output vocabulary.
            input_channels (int): Number of input channels (e.g., 3 for RGB, 1 for grayscale).
            hidden_size (int): Number of hidden units in the LSTM layers.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout probability for LSTM and fully connected layers.
            sequence_length (int, optional): Desired sequence length. If None, it will be dynamic based on input width.
        """
        super(OCRModel, self).__init__()

        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.adaptive_pool = None
        if sequence_length is not None:
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, sequence_length))

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).
        
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, vocab_size).
        """
        x = self.feature_extractor(x)

        if self.adaptive_pool is not None:
            x = self.adaptive_pool(x)

        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, width, -1)

        x, _ = self.lstm(x)
        x = self.classifier(x)

        return x