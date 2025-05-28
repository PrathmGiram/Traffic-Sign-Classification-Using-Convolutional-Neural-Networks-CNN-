from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def get_loaders(data_dir, batch_size=32):
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Paths
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")

    # Load training dataset
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)

    # If validation folder doesn't exist or is empty, split train into train+val
    if not os.path.exists(val_path) or len(os.listdir(val_path)) == 0:
        print("Validation folder not found. Splitting training data 80/20...")
        total_size = len(train_dataset)
        val_size = int(0.2 * total_size)
        train_size = total_size - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        val_dataset.dataset.transform = val_transform  # apply val transform
    else:
        # Load validation dataset from val folder
        val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset.dataset.classes  # classes for label names

