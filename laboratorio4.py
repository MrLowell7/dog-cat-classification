# ============================================================================
# MT3006 - LABORATORIO 4
# ----------------------------------------------------------------------------
# Clasificación de imágenes de perros (0) y gatos (1) con un perceptrón simple
# ============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy import io

# ============================================================================
# 1. CARGA DE DATOS
# ============================================================================
data = io.loadmat(r"C:\Users\mt3006lab4\dog_cat_data.mat")
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']

X_train = X_train / 255.0
X_val = X_val / 255.0


# Visualizar dos ejemplos
example1 = np.reshape(X_train[19, :], (32,32), order='F')
example2 = np.reshape(X_train[20, :], (32,32), order='F')
plt.subplot(121)
plt.imshow(example1, cmap='gray')
plt.title(f'Categoría: {y_train[19,0]}')
plt.subplot(122)
plt.imshow(example2, cmap='gray')
plt.title(f'Categoría: {y_train[20,0]}')
plt.show()

# Convertir a tensores
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# ============================================================================
# 2. DATA LOADERS
# ============================================================================
batch_size = 20
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# ============================================================================
# 3. DEFINICIÓN DEL MODELO (PERCEPTRÓN SIMPLE)
# ============================================================================
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024, 1)  # 32x32=1024 entradas

    def forward(self, x):
        x = self.fc(x)
        return torch.sigmoid(x)

model = Perceptron()

# ============================================================================
# 4. PÉRDIDA Y OPTIMIZADOR
# ============================================================================
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.99, weight_decay=0.001)

# ============================================================================
# 5. CICLO DE ENTRENAMIENTO
# ============================================================================
num_epochs = 50
train_losses, val_losses = [], []
train_acc, val_acc = [], []

for epoch in range(num_epochs):
    # Entrenamiento
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (outputs >= 0.5).float()
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    train_losses.append(running_loss / len(train_loader))
    train_acc.append(correct / total)

    # Validación
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            outputs = model(xb)
            loss = criterion(outputs, yb)
            running_loss += loss.item()
            preds = (outputs >= 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    val_losses.append(running_loss / len(val_loader))
    val_acc.append(correct / total)

    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
          f"Train Acc: {train_acc[-1]:.2f}, Val Acc: {val_acc[-1]:.2f}")

print("Entrenamiento terminado")

# ============================================================================
# 6. GRAFICAR HISTÓRICOS
# ============================================================================
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title("Evolución de la pérdida")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(train_acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.legend()
plt.title("Evolución de la exactitud")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# ============================================================================
# 7. MATRIZ DE CONFUSIÓN
# ============================================================================
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        outputs = model(xb)
        preds = (outputs >= 0.5).int()
        all_preds.extend(preds.numpy().flatten())
        all_labels.extend(yb.numpy().flatten())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Perro", "Gato"])
disp.plot(cmap=plt.cm.Blues)
plt.show()

# ============================================================================
# 8. VISUALIZAR PESOS DEL MODELO - BLANCO Y NEGRO
# ============================================================================
from skimage.filters import gaussian
import matplotlib.patches as mpatches

# Obtener pesos y remodelar
weights = model.fc.weight.detach().numpy().reshape(32, 32, order='F')

# Suavizado con filtro gaussiano
weights_smooth = gaussian(weights, sigma=(2, 2), truncate=2)

# Crear figura con dos subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Pesos originales - Blanco y Negro
# Los pesos positivos (gatos) en blanco, los negativos (perros) en negro
im1 = axs[0].imshow(weights, cmap='gray')
axs[0].set_title("Pesos aprendidos (originales)")
fig.colorbar(im1, ax=axs[0])
axs[0].set_xlabel("Píxeles blancos = Gato, Píxeles negros = Perro")

# Pesos suavizados - Blanco y Negro
im2 = axs[1].imshow(weights_smooth, cmap='gray')
axs[1].set_title("Pesos aprendidos (suavizados)")
fig.colorbar(im2, ax=axs[1])
axs[1].set_xlabel("Píxeles blancos = Gato, Píxeles negros = Perro")

plt.tight_layout()
plt.show()

# Explicación adicional en texto
print("\nINTERPRETACIÓN DE LOS PESOS:")
print("- Las zonas BLANCAS indican características que sugieren GATO")
print("- Las zonas NEGRAS indican características que sugieren PERRO")
print("- Las zonas GRISES son características neutras o menos importantes")