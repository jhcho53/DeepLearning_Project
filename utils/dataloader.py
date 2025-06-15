import os
import random

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

class VOCDataset(Dataset):
    def __init__(self, root: str, image_set: str = "train", train: bool = True):
        voc_root = os.path.join(root, "VOCdevkit", "VOC2012")
        list_path = os.path.join(
            voc_root, "ImageSets", "Segmentation", f"{image_set}.txt"
        )
        with open(list_path, "r") as f:
            self.ids = [x.strip() for x in f]

        self.img_dir  = os.path.join(voc_root, "JPEGImages")
        self.mask_dir = os.path.join(voc_root, "SegmentationClass")
        self.train    = train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id   = self.ids[idx]
        img_path = os.path.join(self.img_dir,  img_id + ".jpg")
        msk_path = os.path.join(self.mask_dir, img_id + ".png")

        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(msk_path)

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

        img  = TF.to_tensor(img)
        img  = TF.normalize(img, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return img, mask

def get_voc_loaders(
    root: str,
    batch_size: int,
    num_workers: int,
    train_set: str = "train",
    val_set: str   = "val",
):
    train_ds = VOCDataset(root, image_set=train_set, train=True)
    val_ds   = VOCDataset(root, image_set=val_set,   train=False)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
