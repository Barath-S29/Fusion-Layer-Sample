# src/dataset.py
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

def get_transforms(img_size=224, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])

def get_dataloaders(data_dir, batch_size=32, img_size=224, num_workers=4):
    train_dir = os.path.join(data_dir, 'train')
    val_dir   = os.path.join(data_dir, 'val')
    test_dir  = os.path.join(data_dir, 'test')

    if not os.path.exists(val_dir):           #Added these lines to avoid errors if val/test dirs are missing
        val_dir = train_dir
    if not os.path.exists(test_dir):
        test_dir = train_dir

    train_ds = datasets.ImageFolder(train_dir, transform=get_transforms(img_size, True))
    val_ds   = datasets.ImageFolder(val_dir,   transform=get_transforms(img_size, False))
    test_ds  = datasets.ImageFolder(test_dir,  transform=get_transforms(img_size, False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
