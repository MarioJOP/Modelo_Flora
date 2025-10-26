import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

def build_model(base_name: str, num_classes: int, device: torch.device, dropout=0.2):
    """
    Cria um modelo de classificação baseado em uma rede pré-treinada (transfer learning).

    Args:
        base_name (str): Nome da rede base ('vgg16', 'resnet50', 'mobilenetv2', etc.)
        num_classes (int): Número de classes de saída.
        device (torch.device): CPU ou GPU onde o modelo será executado.

    Returns:
        torch.nn.Module: Modelo completo pronto para treino.
    """

    base_name = base_name.lower()

    if base_name == "vgg16":
        base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for param in base_model.features.parameters():
            param.requires_grad = False

        base_model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    elif base_name == "resnet50":
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in base_model.parameters():
            param.requires_grad = False

        in_features = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    elif base_name == "mobilenetv2":
        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        for param in base_model.features.parameters():
            param.requires_grad = False

        in_features = base_model.classifier[1].in_features
        base_model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    else:
        raise ValueError(f"Modelo base '{base_name}' não suportado.")

    # Enviar para GPU/CPU
    return base_model.to(device)

def save_model(model, optimizer, epoch, val_acc, output_dir='models', prefix='vgg16'):
    """
    Salva o modelo, otimizador e metadados de treinamento.

    Args:
        model (torch.nn.Module): O modelo treinado.
        optimizer (torch.optim.Optimizer): O otimizador usado no treino.
        epoch (int): Última época concluída.
        val_acc (float): Acurácia de validação (para nome do arquivo).
        output_dir (str): Diretório onde o modelo será salvo.
        prefix (str): Prefixo no nome do arquivo (ex: nome do modelo).
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{prefix}_epoch{epoch}_val{val_acc:.4f}.pth"
    path = os.path.join(output_dir, filename)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, path)

    print(f"✅ Modelo salvo em: {path}")