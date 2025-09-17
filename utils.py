import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from glob import glob
import os
from dataset import BraTSDataset


def get_data_paths(processed_dir):
    """get a list of all preprocessed subject file paths"""
    return sorted(glob(os.path.join(processed_dir, '*_img.npy')))


def get_loaders(processed_dir, batch_size, test_size=0.2):
    """splits data into train val set and return dataloaders"""
    all_files = get_data_paths(processed_dir)
    train_files, val_files = train_test_split(all_files, test_size=test_size, random_state=42)

    train_dataset = BraTSDataset(data_dir=None, patch_size=(64, 64, 64), is_train=True)
    train_dataset.data_files = train_files 

    val_dataset = BraTSDataset(data_dir=None, patch_size=(64, 64, 64), is_train=False)
    val_dataset.data_files = val_files

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader


def save_checkpoint(state, filename="brats_checkpoint.pth.tar"):
    print("Saving Checkpoint----")
    torch.save(state, filename)


def dice_score(preds, targets, smooth=1e-6):
    preds = torch.softmax(preds, dim=1)
    # find the class with highest probab for each pixel
    preds = torch.argmax(preds, dim=1)

    dice_scores = []
    # calculate dicescore for each class
    for i in range(1, 4):
        pred_i = (preds==i).float()
        target_i = (targets==i).float()
        intersection = (pred_i * target_i).sum()
        union = pred_i.sum() + target_i.sum()
        dice = (2 *intersection + smooth) / (union + smooth)
        dice_scores.append(dice)

    return torch.mean(torch.tensor(dice_scores))

