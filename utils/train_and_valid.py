import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(dataloader, desc=f"Treinando época: {epoch+1}", total=len(dataloader)):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, loss_fn, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Validando época: {epoch+1}", total=len(dataloader)):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc

def train_model(train_loader, val_loader, model, loss_fn, optimizer, device, num_epochs=10):
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc': []
    }

    for epoch in tqdm(range(num_epochs), desc="Treinando o modelo", total=num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        val_loss,   val_acc   = validate(model,   val_loader,   loss_fn, device, epoch)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}\n"
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}\n"
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}\n")

    return history
