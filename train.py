import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50

from loss.loss import WeightedFocalDiceLoss
from utils.viz import plot_losses
from utils.dataloader import get_voc_loaders

# 새로운 학습 스크립트: Sobel 기반 엣지 손실 제거, Shape-Aware Boundary Loss 통합

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   type=str, default="/path/to/VOCdevkit/VOC2012")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--workers",     type=int,   default=4)
    parser.add_argument("--save_dir",    type=str,   default="./checkpoints")
    parser.add_argument("--shape_weight",type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_voc_loaders(
        args.data_root, args.batch_size, args.workers
    )

    # segmentation backbone
    model = deeplabv3_resnet50(pretrained=False, num_classes=7).to(device)

    # loss: Focal+Dice + Shape-Aware Boundary Loss
    class_weights = torch.tensor([3.,7.,1.,1.,1.,4.,4.], device=device)
    seg_loss_fn = WeightedFocalDiceLoss(
        alpha=class_weights, gamma=2.0,
        dice_weight=3.0,
        shape_weight=args.shape_weight,
        ignore_index=255
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float("inf")

    train_losses     = []
    val_losses       = []
    train_edge_losses = []
    val_edge_losses   = []

    for epoch in range(1, args.epochs + 1):
        # --- Training ---
        model.train()
        epoch_seg_loss  = 0.0
        epoch_edge_loss = 0.0
        for batch_idx, (imgs, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            logits = model(imgs)["out"]  # (B,7,H,W)
            loss, focal_l, dice_l, shape_l = seg_loss_fn(logits, masks)

            loss.backward()
            optimizer.step()

            epoch_seg_loss  += loss.item()
            epoch_edge_loss += shape_l.item()
            print(f"Batch {batch_idx}: total={loss.item():.4f}, edge={shape_l.item():.4f}")

        train_losses.append(epoch_seg_loss  / len(train_loader))
        train_edge_losses.append(epoch_edge_loss / len(train_loader))

        # --- Validation ---
        model.eval()
        val_seg_loss  = 0.0
        val_edge_loss = 0.0
        with torch.no_grad():
            for batch_idx, (imgs, masks) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)):
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)["out"]
                loss, _, _, shape_l = seg_loss_fn(logits, masks)
                val_seg_loss  += loss.item()
                val_edge_loss += shape_l.item()
                print(f"Val Batch {batch_idx}: seg={loss.item():.4f}, edge={shape_l.item():.4f}")

        val_losses.append(val_seg_loss  / len(val_loader))
        val_edge_losses.append(val_edge_loss / len(val_loader))

        scheduler.step()
        print(f"[{epoch}/{args.epochs}] Train Seg: {train_losses[-1]:.4f} |"
            f" Train Edge: {train_edge_losses[-1]:.4f} | "
            f"Val Seg: {val_losses[-1]:.4f} | Val Edge: {val_edge_losses[-1]:.4f}")

        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            torch.save(model.state_dict(),
                    os.path.join(args.save_dir, f"best_epoch{epoch}.pth"))

    # --- 2) 마지막에 plot 호출
    save_path = os.path.join(args.save_dir, "loss_curve.png")
    plot_losses(
        train_losses,
        val_losses,
        train_edge_losses,
        val_edge_losses,
        save_path
    )


if __name__ == "__main__":
    main()
