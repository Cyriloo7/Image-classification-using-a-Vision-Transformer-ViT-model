import argparse
import os
import shutil
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
from model import build_vit_prediction


def predict_folder(folder_path: str, ckpt_path: str, img_size: int = 224, out_dir: str = "predicted_results"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define classes (make sure order matches training!)
    classes = ['Simple', 'complicated']

    # Load model
    model = build_vit_prediction(len(classes), ckpt_path)
    model.to(device).eval()

    # Transform
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])

    folder_path = os.path.abspath(folder_path)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Loop through images in folder
    for fname in tqdm(os.listdir(folder_path), desc="Predicting"):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(folder_path, fname)
        img = Image.open(img_path).convert('RGB')
        x = tf(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            idx = logits.argmax(dim=1).item()

        pred_class = classes[idx]

        # Save/copy to predicted folder
        class_dir = os.path.join(out_dir, pred_class)
        os.makedirs(class_dir, exist_ok=True)
        shutil.copy(img_path, os.path.join(class_dir, fname))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help="Path to a folder containing images")
    parser.add_argument('--ckpt', required=True, help="Path to checkpoint file")
    parser.add_argument('--out_dir', default="predicted_results", help="Where to save sorted predictions")
    args = parser.parse_args()

    predict_folder(args.image, args.ckpt, out_dir=args.out_dir)
