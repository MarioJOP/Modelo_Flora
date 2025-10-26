import torch
import torch.optim as optim
from utils.train_and_valid import train_model
from utils.model import  build_model
from utils.data import create_data_loaders
from utils.plot_history import plot_history
import torch.nn as nn
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_name = "resnet50"
data_dir = Path("dados")

train_dir = data_dir / "dados_separados" / "train"
val_dir = data_dir / "dados_separados" / "val"

train_loader, val_loader, classes = create_data_loaders(train_dir, val_dir, batch_size=32)
model = build_model(base_name=base_name, num_classes=len(classes), device=device)

loss_fn = nn.CrossEntropyLoss()

optimizers = {
    'Adam':   optim.Adam,
    # 'SGD':    optim.SGD,
    # 'RMSprop':optim.RMSprop
}
learning_rates = [1e-4] #, 1e-4, 1e-5]
results = {}

for opt_name, opt_class in optimizers.items():
    results[opt_name] = {}
    for lr in learning_rates:
        print(f"\n*** Treinando com {opt_name}, lr={lr}")
        # recriar modelo fresh para cada combinação
        model = build_model(base_name=base_name, num_classes=len(classes), device=device, dropout=0.3)
        optimizer = opt_class(model.parameters(), lr=lr)
        history = train_model(train_loader, val_loader, model, loss_fn, optimizer, device, num_epochs=50)
        results[opt_name][lr] = history
        plot_history(history, opt_name, lr)