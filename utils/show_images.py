import matplotlib.pyplot as plt
import torch

def show_random_images(train_loader, classes, num_images=6):
    # Pega um batch aleatório
    images, labels = next(iter(train_loader))

    # Seleciona algumas imagens do batch
    num_images = min(num_images, len(images))
    images = images[:num_images]
    labels = labels[:num_images]

    # Desfaz normalização para exibir cores reais
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])

    # Cria uma figura
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    if num_images == 1:
        axes = [axes]

    for i in range(num_images):
        img = images[i].permute(1, 2, 0)  # [C,H,W] → [H,W,C]
        img = img * std + mean             # desfaz normalização
        img = torch.clamp(img, 0, 1)       # garante faixa [0,1]

        axes[i].imshow(img)
        axes[i].set_title(classes[labels[i]])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
