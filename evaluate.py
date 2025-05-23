import argparse
import torch
import torch.nn as nn
from data_utils import get_orientation_loaders_evaluation
from model import build_vit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model_ckpt', required=True)
    args = parser.parse_args()

    # data_dir = r'data'
    # model_ckpt = r'models/best_vit.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_loader, classes = get_orientation_loaders_evaluation(args.data_dir, batch_size=4)

    model = build_vit(len(classes), args.model_ckpt)
    model.load_state_dict(torch.load(args.model_ckpt, map_location=device))
    model.to(device).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu()
            y_true += labels.tolist()
            y_pred += preds.tolist()

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))
    # Plot the confusion matrix
    cm_display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.show()

if __name__ == '__main__':
    main()