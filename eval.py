import os
import time
import torch
import numpy as np
from PIL import Image
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50
from torchvision import transforms
from tqdm import tqdm

# 클래스 팔레트 정의 (RGB 값 순서로 나열)
CUSTOM_PALETTE = [
    128,   0, 128,   # 0 road (purple)
    255, 192, 203,   # 1 sidewalk (pink)
    255, 255,   0,   # 2 object (yellow)
      0, 255,   0,   # 3 vegetation (green)
      0, 191, 255,   # 4 sky (sky blue)
    255,   0,   0,   # 5 human (red)
      0,   0, 128,   # 6 vehicle (navy)
]

NUM_CLASSES = 7

def fast_confusion_matrix(pred, target, num_classes):
    mask = (target >= 0) & (target < num_classes)
    return torch.bincount(
        num_classes * target[mask].to(torch.int64) + pred[mask],
        minlength=num_classes**2
    ).reshape(num_classes, num_classes).cpu().numpy()

def compute_miou(conf_matrix):
    intersection = np.diag(conf_matrix)
    union = conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)
    iou = intersection / (union + 1e-6)
    return iou, np.mean(iou)

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])
    return transform(img)

def evaluate_whole(model_path, voc_root, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dir = os.path.join(voc_root, "JPEGImages")
    mask_dir  = os.path.join(voc_root, "SegmentationClass")
    val_list  = os.path.join(voc_root, "ImageSets/Segmentation/val.txt")

    model = deeplabv3_resnet50(pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    os.makedirs(save_dir, exist_ok=True)
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_frames = 0
    start_time = time.time()

    with open(val_list, "r") as f:
        val_ids = [line.strip() for line in f.readlines()]

    with torch.no_grad():
        for img_id in tqdm(val_ids, desc="Evaluating full-size"):
            img_path = os.path.join(image_dir, img_id + ".jpg")
            mask_path = os.path.join(mask_dir, img_id + ".png")

            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path)

            input_tensor = preprocess_image(img).unsqueeze(0).to(device)
            gt_mask = torch.from_numpy(np.array(mask)).to(device)

            output = model(input_tensor)["out"]
            pred = output.argmax(dim=1).squeeze(0).to(torch.uint8)

            conf_matrix += fast_confusion_matrix(pred, gt_mask, NUM_CLASSES)

            # ⬇️ 팔레트 기반 저장
            color_mask = Image.fromarray(pred.cpu().numpy(), mode="P")
            color_mask.putpalette(CUSTOM_PALETTE)
            save_path = os.path.join(save_dir, img_id + ".png")
            color_mask.save(save_path)

            total_frames += 1

    end_time = time.time()
    ious, miou = compute_miou(conf_matrix)
    fps = total_frames / (end_time - start_time)

    print("\nEvaluation Results:")
    print(f"mIoU: {miou:.4f}")
    for i, iou in enumerate(ious):
        print(f" - Class {i}: IoU = {iou:.4f}")
    print(f"FPS: {fps:.2f} frames/sec")
    print(f"Saved color masks to {save_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_root",   type=str, required=True, help="Path to VOC2012 root")
    parser.add_argument("--save_dir",   type=str, default="./val_preds_full")
    args = parser.parse_args()

    evaluate_whole(args.model_path, args.data_root, args.save_dir)
