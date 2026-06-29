import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir='data', batch_size=64, img_size=224):
    """
    Carga el dataset FER-2013, aplica transformaciones y devuelve los DataLoaders.
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), # ResNet18 waits for RGB
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=test_dir, transform=val_transforms)

    # Crear los DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_dataset.classes

# Local testing of the DataLoaders
if __name__ == "__main__":
    print("Probando DataLoaders...")
    train_loader, val_loader, classes = get_dataloaders()
    print(f"Clases encontradas: {classes}")
    images, labels = next(iter(train_loader))
    print(f"Formato del batch de imágenes: {images.shape}")
    print(f"Formato del batch de etiquetas: {labels.shape}")