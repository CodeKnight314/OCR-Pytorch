from model import OCRModel
from dataset import load_dataset, DataLoader
from loss import OCRLoss
import argparse
import pandas as pd
import torch
from tqdm import tqdm
import torch.optim as opt
import os
import logging

def train_OCR(model: OCRModel, criterion: OCRLoss, train_dl: DataLoader, val_dl: DataLoader, lr: float, epochs: int, save_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    optimizer = opt.AdamW(model.parameters(), lr=lr)
    scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=lr * 0.01)

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
        model.eval()
        logging.info(f"Epoch {epoch}/{epochs} - Validation")
        for image, label in tqdm(val_dl, desc=f"Validation Epoch {epoch}"):
            image, label = image.to(device), label.to(device)

            with torch.no_grad():
                prediction = model(image)

            loss = criterion(prediction, label)
            total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dl)
        logging.info(f"Epoch {epoch} - Validation Loss: {avg_val_loss:.4f}")

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
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--save_dir", type=str, default="checkpoints/", help="Directory to save model checkpoints")
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    train_dl = load_dataset(root_dir=args.root, mode="train", csv=args.csv, size=(64, 128), batch_size=args.batch_size)
    val_dl = load_dataset(root_dir=args.root, mode="val", csv=args.csv, size=(64, 128), batch_size=args.batch_size)

    df = pd.read_csv(args.csv)["labels"]
    all_characters = ''.join(df.astype(str))
    unique_characters = set(all_characters)

    model = OCRModel(vocab_size=len(unique_characters))
    criterion = OCRLoss(blank=len(unique_characters), pad=0, ctc_weight=1.0)

    train_OCR(
        model=model,
        criterion=criterion,
        train_dl=train_dl,
        val_dl=val_dl,
        lr=args.lr,
        epochs=args.epochs,
        save_dir=args.save_dir
    )
