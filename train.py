import torch
import torch.optim as optim
from tqdm import tqdm
from model import UNet3D

from utils import get_loaders, save_checkpoint, dice_score
from monai.losses import DiceCELoss


LEARNING_RATE = 1e-4
BATCH_SIZE = 1
EPOCHS = 200
PROCESSED_DATA_PATH = "BRATS-data/processed"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, desc="Training")

    for batch_idx, (data, tagets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE).long()
        
        # forward propagation
        with torch.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward propagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())


def validate_fn(loader, model, loss_fn):
    model.eval()
    val_loss = 0
    val_dice = 0
    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Validation"):
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE).long()

            predictions = model(data)
            loss = loss_fn(predictions, targets)
            val_loss += loss.item()

            dice = dice.score(predictions, targets)
            val_dice += dice.item()
    
    avg_loss = val_loss / len(loader)
    avg_dice = val_dice / len(loader)

    print(f"Validation Avg Loss: {avg_loss:.3f}, Avg Dice: {avg_dice:.4f}")
    model.train()
    return avg_dice


def main():
    model = UNet3D(in_channels=4, out_channels=4).to(DEVICE)
    loss_fn = DiceCELoss(to_onehot_y=False, softmax=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.GradScaler()

    train_loader, val_loader = get_loaders(PROCESSED_DATA_PATH, BATCH_SIZE)
    best_dice_score = -1

    for epoch in range(EPOCHS):
        print(f"------Epoch {epoch+1}/{EPOCHS}--------")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        current_dice = validate_fn(val_loader, model, loss_fn)

        # save the best checkpoint
        if current_dice > best_dice_score:
            best_dice_score = current_dice
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename="best_model.pth.tar")


if __name__ == "__main__":
    main()

