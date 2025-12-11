from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_data_loaders(train_dir, val_dir, batch_size=32, target_size=(224,224)):
    # Transformações
    train_transforms = transforms.Compose([
        # O crop deve ser mais agressivo para pegar detalhes (folhas/casca)
        transforms.RandomResizedCrop(target_size, scale=(0.4, 1.0)), 
        
        # Rotação e Flip são essenciais para objetos naturais
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30), # Aumentei um pouco a liberdade
        
        # Variação de cor é crucial para ambientes externos (sol/sombra)
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3)
        ], p=0.1),

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        # Resize para um pouco maior que o target e depois crop central
        # Isso preserva a proporção da árvore melhor que apenas Resize(target_size)
        transforms.Resize(256), 
        transforms.CenterCrop(target_size),
        
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    print("Tamanho do dataset de Treino:", len(train_dataset))
    print("Tamanho do dataset de Validação:", len(val_dataset))

    return train_loader, val_loader, train_dataset.classes