import os
import random
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from PIL import Image
import numpy as np
from tqdm import tqdm

class VOCDataset(Dataset):
    def __init__(self, root, image_set="train", train=True):
        voc_root = os.path.join(root, "VOCdevkit", "VOC2012")
        list_path = os.path.join(voc_root, "ImageSets", "Segmentation", f"{image_set}.txt")
        with open(list_path, "r") as f:
            self.ids = [x.strip() for x in f]
        self.img_dir  = os.path.join(voc_root, "JPEGImages")
        self.mask_dir = os.path.join(voc_root, "SegmentationClass")
        self.train = train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path  = os.path.join(self.img_dir,  img_id + ".jpg")
        mask_path = os.path.join(self.mask_dir, img_id + ".png")

        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # 공간 변형: train일 때만 랜덤 크롭/뒤집기, val은 단일 리사이즈
        if self.train:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img, scale=(0.5, 1.0), ratio=(0.75, 1.333)
            )
            img  = TF.resized_crop(img,  i, j, h, w, size=(512,512), interpolation=Image.BILINEAR)
            mask = TF.resized_crop(mask, i, j, h, w, size=(512,512), interpolation=Image.NEAREST)
            if random.random() > 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)
        else:
            img  = TF.resize(img,  (512,512), interpolation=Image.BILINEAR)
            mask = TF.resize(mask, (512,512), interpolation=Image.NEAREST)

        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return img, mask

def get_voc_loaders(root, batch_size, num_workers):
    train_ds = VOCDataset(root, image_set="train", train=True)
    val_ds   = VOCDataset(root, image_set="val",   train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--batch_size",type=int, default=8)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--save_dir",  type=str, default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_voc_loaders(args.data_root, args.batch_size, args.workers)

    model = deeplabv3_mobilenet_v3_large(pretrained=False, num_classes=7)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float("inf")
    for epoch in range(1, args.epochs+1):
        # Train
        model.train()
        t_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            out = model(imgs)["out"]
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)

        # Validate
        model.eval()
        v_loss = 0
        for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)["out"]
            v_loss += criterion(out, masks).item()
        v_loss /= len(val_loader)
        scheduler.step()

        print(f"[{epoch}/{args.epochs}] train: {t_loss:.4f}, val: {v_loss:.4f}")

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f"best_epoch{epoch}.pth"))

if __name__ == "__main__":
    main()