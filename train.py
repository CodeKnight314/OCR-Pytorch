from model import OCRModel
from dataset import load_dataset, DataLoader
from loss import OCRLoss, LevenShteinDistance
import argparse
import pandas as pd
import torch
from tqdm import tqdm
import torch.optim as opt
import os
import logging
import numpy as np
import json

def decode_logits_output(prediction: torch.Tensor, idx2char: dict):
    """
    Args: 
        prediction (torch.Tensor): Predicted logits tensor with shape [batch_size, seq_len, vocab_size] 
    """
    prediction = prediction.argmax(dim=-1)
    batch_string = []
    for sequence in prediction: 
        text = ''.join(idx2char[idx] for idx in np.array(sequence.cpu()))
        text = ''.join(char for char in text if char != idx2char[0])
        batch_string.append(text)
    return batch_string

def decode_label(label: torch.Tensor, idx2char: dict): 
    """
    Args: 
        label (torch.Tensor): Label tensor with shape [batch_size, max_len]
    """
    batch_string = [] 
    for sequence in label: 
        text = ''.join(idx2char[idx] for idx in np.array(sequence.cpu()))
        text = ''.join(char for char in text if char != idx2char[0])
        batch_string.append(text)
    return batch_string

def train_OCR(model: OCRModel, 
              criterion: OCRLoss, 
              train_dl: DataLoader, 
              val_dl: DataLoader, 
              lr: float, 
              epochs: int, 
              save_dir: str, 
              char_dict: dict, 
              idx_dict: dict):
    """
    Args:
        model (OCRModel): The OCR model to be trained.
        criterion (OCRLoss): Loss function to optimize the model.
        train_dl (DataLoader): DataLoader for the training dataset.
        val_dl (DataLoader): DataLoader for the validation dataset.
        lr (float): Learning rate for the optimizer.
        epochs (int): Number of epochs for training.
        save_dir (str): Directory path to save model checkpoints.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    optimizer = opt.AdamW(model.parameters(), lr=lr)
    scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=int(epochs * 0.75), eta_min=lr * 0.01)

    logging.info(f"Training started on {device}")

    for epoch in range(1, epochs + 1):
        total_tr_loss = 0.0
        model.train()
        logging.info(f"Epoch {epoch}/{epochs} - Training")
        for image, label in tqdm(train_dl, desc=f"Training Epoch {epoch}"):
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()
            prediction = model(image)

            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()

            total_tr_loss += loss.item()

        avg_tr_loss = total_tr_loss / len(train_dl)
        logging.info(f"Epoch {epoch} - Training Loss: {avg_tr_loss:.4f}")

        total_val_loss = 0.0
        total_lv_loss = 0.0
        model.eval()
        logging.info(f"Epoch {epoch}/{epochs} - Validation")
        for image, label in tqdm(val_dl, desc=f"Validation Epoch {epoch}"):
            image, label = image.to(device), label.to(device)

            with torch.no_grad():
                prediction = model(image)
            
            pred_ls = decode_logits_output(prediction, idx_dict)
            grth_ls = decode_label(label, idx_dict)
            
            levn_dis = [] 
            for pred, grth in zip(pred_ls, grth_ls): 
                levn_dis.append(LevenShteinDistance(pred, grth))
            total_lv_loss += sum(levn_dis)/len(levn_dis)

            loss = criterion(prediction, label)
            total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dl)
        avg_lev_loss = total_lv_loss / len(val_dl)
        logging.info(f"Epoch {epoch} - Validation Loss: {avg_val_loss:.4f}")
        logging.info(f"Epoch {epoch} - Levenshtein Loss: {avg_lev_loss:4f}")

        scheduler.step()

        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Saved checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an OCR model")
    parser.add_argument("--root", type=str, required=True, help="Root directory for the dataset")
    parser.add_argument("--csv", type=str, required=True, help="Path to the CSV file containing labels")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--save_dir", type=str, default="checkpoints/", help="Directory to save model checkpoints")
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    train_dl = load_dataset(root_dir=args.root, mode="train", csv=args.csv, size=(64, 128), batch_size=args.batch_size)
    val_dl = load_dataset(root_dir=args.root, mode="val", csv=args.csv, size=(64, 128), batch_size=args.batch_size)

    df = pd.read_csv(args.csv)
    all_chars = ''.join(df["labels"].astype(str))
    unique_chars = sorted(set(all_chars))
    
    char_dict = {'<blank>': 0, **{c: i+1 for i, c in enumerate(unique_chars)}}
    idx_dict = {v: k for k, v in char_dict.items()}

    with open('char_idx_dicts.json', 'w') as f:
        json.dump({'char_dict': char_dict, 'idx_dict': idx_dict}, f)

    model = OCRModel(vocab_size=len(char_dict))
    criterion = OCRLoss(blank=0, pad=-1, ctc_weight=1.0)

    train_OCR(
        model=model,
        criterion=criterion,
        train_dl=train_dl,
        val_dl=val_dl,
        lr=args.lr,
        epochs=args.epochs,
        save_dir=args.save_dir,
        char_dict=char_dict, 
        idx_dict=idx_dict
    )
