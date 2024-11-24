import torch
import torch.nn as nn
import random

class OCRLoss(nn.Module): 
    def __init__(self, blank: int, pad: int, ctc_weight: float): 
        super().__init__()
        
        self.ctc_weight = ctc_weight
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=True)
        self.pad = pad
        
    def compute_CTC_Loss(self, prediction: torch.Tensor, targets: torch.Tensor, pred_len: torch.Tensor, target_len: torch.Tensor):
        """
        Args: 
            prediction (torch.Tensor): log-softmaxed tensor with defined shape of [sequence_length, batch_size, vocab_size].
            targets (torch.Tensor): target tensor with defined shape of [batch_size, max_sequence_length].
            pred_len (torch.Tensor): Lengths of predictions (shape: [batch_size]).
            target_len (torch.Tensor): Lengths of targets (shape: [batch_size]).
        
        Returns: 
            torch.Tensor: CTC Loss comparing two sequences.
        """
        if prediction.shape[0] == targets.shape[0]:
            prediction = prediction.transpose(0, 1)
        loss = self.ctc_loss(prediction, targets, pred_len, target_len)
        return loss
    
    def forward(self, prediction: torch.Tensor, targets: torch.Tensor): 
        """
        Args: 
            prediction (torch.Tensor): Log-softmaxed tensor with shape [batch_size, sequence_length, vocab_size].
            targets (torch.Tensor): Target tensor with shape [batch_size, max_sequence_length].
        
        Returns: 
            torch.Tensor: Combined Loss for sequence prediction.
        """
        batch_size, seq_len, _ = prediction.shape
        batch_size, max_len = targets.shape

        prediction_length = torch.full(
            size=(batch_size,), fill_value=seq_len, dtype=torch.long, device=prediction.device
        )

        target_length = torch.sum(targets != self.pad, dim=1)

        return self.ctc_weight * self.compute_CTC_Loss(prediction, targets, prediction_length, target_length)
    
def LevenShteinDistance(prediction: str, ground_truth: str):
    m = len(prediction)
    n = len(ground_truth)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if prediction[i - 1] == ground_truth[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # deletion
                    dp[i][j - 1] + 1,  # insertion
                    dp[i - 1][j - 1] + 1  # substitution
                )

    return dp[m][n]

if __name__ == "__main__": 
    batch_size = 16
    sequence_len = 32 
    vocab_size = 34
    min_seq_len = 3 
    max_seq_len = 12

    x = [torch.randint(1, vocab_size, (random.randint(min_seq_len, max_seq_len),)) for _ in range(batch_size)]
    x = [torch.nn.functional.pad(x[i], (0, max_seq_len - x[i].shape[0]), value=0) for i in range(len(x))]
    targets = torch.stack(x)

    prediction = torch.rand((batch_size, sequence_len, vocab_size), dtype=torch.float32)

    module = OCRLoss(blank=vocab_size - 1, pad=0, ctc_weight=1.0)

    loss = module(prediction, targets)
    print(f"Loss: {loss.item()}")