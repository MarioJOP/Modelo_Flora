import matplotlib.pyplot as plt
import matplotlib

def plot_history(history, optim_name, lr, filename=None):
    plt.figure(figsize=(12,8))
    # accuracy
    plt.subplot(2,1,1)
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'],   label='val_acc')
    plt.title(f'Accuracy – {optim_name} lr={lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # loss
    plt.subplot(2,1,2)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'],   label='val_loss')
    plt.title(f'Loss – {optim_name} lr={lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()

    if filename:
        filename = filename or "plot_history.png"
        plt.savefig(filename)
        print(f"📊 Gráfico salvo em {filename}")
        plt.close()
    else:
        plt.show()

