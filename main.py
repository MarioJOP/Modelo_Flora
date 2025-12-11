import torch
import torch.optim as optim
from utils.train_and_valid import train_model
from utils.model import  build_model
from utils.data import create_data_loaders
from utils.plot_history import plot_history
import torch.nn as nn
from pathlib import Path
from utils.model import save_model, unfreeze_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_name = "resnet50"
data_dir = Path("dados")

train_dir = Path("dados_separados") / "train"
val_dir = Path("dados_separados") / "val"

results_path = Path("results")
results_path.mkdir(parents=True, exist_ok=True)
plots_path = results_path / "plots"
plots_path.mkdir(parents=True, exist_ok=True)

train_loader, val_loader, classes = create_data_loaders(train_dir, val_dir, batch_size=32)

loss_fn = nn.CrossEntropyLoss()

EPOCHS_STAGE_1 = 40   # Apenas treinar a nova "cabeça" (rápido)
EPOCHS_STAGE_2 = 90  # Fine-tuning da rede inteira (lento e cuidadoso)
LR_STAGE_1 = 1e-3    # LR padrão para aprender o classificador inicial
LR_STAGE_2 = 1e-5    # LR muito baixo para não destruir os pesos da ResNet
WEIGHT_DECAY = 1e-4  # Regularização L2 para combater o overfitting

results = {}

print(f"\n🚀 Iniciando Treinamento em 2 Estágios com ResNet50")

# 1. Construir modelo (Base vem congelada por padrão no seu build_model)
model = build_model(base_name="resnet50", num_classes=len(classes), device=device, dropout=0.6)

# --- ESTÁGIO 1: Treinar apenas o Classificador (Warm-up) ---
print(f"\n[Estágio 1] Treinando apenas o classificador por {EPOCHS_STAGE_1} épocas...")

# Otimizador inicial (apenas parâmetros com requires_grad=True serão atualizados)
optimizer = optim.Adam(model.parameters(), lr=LR_STAGE_1, weight_decay=WEIGHT_DECAY)

# Treina Estágio 1
history_stage1 = train_model(
    train_loader, val_loader, model, loss_fn, optimizer, device, num_epochs=EPOCHS_STAGE_1
)

# Salvar checkpoint do estágio 1 (opcional)
save_model(model, optimizer, epoch=EPOCHS_STAGE_1, val_acc=history_stage1['val_acc'][-1], prefix="resnet50_stage1")


# --- ESTÁGIO 2: Fine-Tuning (Rede Completa) ---
print(f"\n[Estágio 2] Descongelando a rede e fazendo Fine-tuning por {EPOCHS_STAGE_2} épocas...")

# 1. Descongelar a rede
model = unfreeze_model(model)

# 2. Re-inicializar o otimizador com LR menor (importante!)
#    Agora ele vê todos os parâmetros da rede.
optimizer = optim.Adam(model.parameters(), lr=LR_STAGE_2, weight_decay=WEIGHT_DECAY)

# Treina Estágio 2
history_stage2 = train_model(
    train_loader, val_loader, model, loss_fn, optimizer, device, num_epochs=EPOCHS_STAGE_2
)

# --- Juntar os históricos para plotar ---
full_history = {
    'train_loss': history_stage1['train_loss'] + history_stage2['train_loss'],
    'train_acc':  history_stage1['train_acc']  + history_stage2['train_acc'],
    'val_loss':   history_stage1['val_loss']   + history_stage2['val_loss'],
    'val_acc':    history_stage1['val_acc']    + history_stage2['val_acc']
}

# Plotar gráfico combinado
plot_history(full_history, "Adam", f"{LR_STAGE_1}->{LR_STAGE_2}", filename=plots_path / f"learning_curve_resnet50_finetuned.png")

# Salvar modelo final
save_model(model, optimizer, epoch=EPOCHS_STAGE_1 + EPOCHS_STAGE_2, val_acc=history_stage2['val_acc'][-1], prefix="resnet50_final")
