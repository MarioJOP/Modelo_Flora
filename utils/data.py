from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_data_loaders(train_dir, val_dir, batch_size=32, target_size=(224,224)):
    # Transformações
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),  # zoom leve
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3)
        ], p=0.1), # simular desfoque de vento ou imprecisão no foco
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    print("Tamanho do dataset de Treino:", len(train_dataset))
    print("Tamanho do dataset de Validação:", len(val_dataset))

    return train_loader, val_loader, train_dataset.classes