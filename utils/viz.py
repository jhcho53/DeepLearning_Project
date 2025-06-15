import matplotlib.pyplot as plt

def plot_losses(train_seg, val_seg, train_edge, val_edge, save_path):
    """
    train_seg, val_seg, train_edge, val_edge: list of loss values per epoch
    save_path: 그림을 저장할 파일 경로 (예: "./checkpoints/loss_curve.png")
    """
    epochs = range(1, len(train_seg) + 1)
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_seg,  label="Train Seg Loss")
    plt.plot(epochs, val_seg,    label="Val   Seg Loss")
    plt.plot(epochs, train_edge, label="Train Edge Loss")
    plt.plot(epochs, val_edge,   label="Val   Edge Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
