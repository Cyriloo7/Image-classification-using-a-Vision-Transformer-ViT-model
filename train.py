import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from model import build_vit
from data_utils import get_orientation_loaders

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(outputs.argmax(dim=1).cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    return avg_loss, acc


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    return avg_loss, acc


def parse_args():
    parser = argparse.ArgumentParser(description="Train Vision Transformer for orientation classification")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset root (with train/ and val/ subfolders)')
    parser.add_argument('--model_out', type=str, default='models/best_vit.pth',
                        help='File path to save the best model checkpoint')
    parser.add_argument('--epochs', type=int, default=4,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='Learning rate for optimizer')
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, classes = get_orientation_loaders(
        args.data_dir,
        batch_size=args.batch_size
    )

    # Compute class weights for cost-sensitive learning
    all_labels = np.array([label for _, label in train_loader.dataset])
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Using class weights: {class_weights}")

    # Build model
    model = build_vit(
        num_classes=len(classes),
        model_out=args.model_out
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch}:"
            f" train_loss={train_loss:.4f}, train_acc={train_acc:.4f};"
            f" val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.model_out)
            print(f"[Checkpoint] New best model (val_acc={val_acc:.4f}) saved to {args.model_out}\n")

if __name__ == '__main__':
    main()
