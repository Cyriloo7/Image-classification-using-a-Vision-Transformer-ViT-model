import argparse
from PIL import Image
import torch
from torchvision import transforms
from model import build_vit_prediction

def predict(image_path: str, ckpt_path: str, img_size: int = 224):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load model
    # assume classes order saved alongside ckpt or same as data_utils
    classes = ['180 degree background', '180 degree skip', 'background', 'skip']
    model = build_vit_prediction(len(classes), ckpt_path)
    model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    img = Image.open(image_path).convert('RGB')
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        idx = logits.argmax(dim=1).item()
    print(f"Prediction: {classes[idx]}")

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--image', required=True)
    # parser.add_argument('--ckpt', required=True)
    # args = parser.parse_args()

    image = r'skip classification\New folder\skip\4d7dba93-4f34-4d72-a028-ee9dc8525353.jpg'
    ckpt = r'models\best_vit.pth'
    predict(image, ckpt)