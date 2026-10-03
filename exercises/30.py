import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
np.random.seed(42)

X, y = load_breast_cancer(return_X_y=True)

# 80/20 test split
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 75/25 validation inside the train part
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25,
    stratify=y_trainval, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

def tensor_data(X, y):
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

Xtr, ytr = tensor_data(X_train, y_train)
Xv, yv = tensor_data(X_val, y_val)
Xte, yte = tensor_data(X_test, y_test)

train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)

model = nn.Sequential(
    nn.Linear(30, 32),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 2)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)

# Warmup Tw=5, total T=100, then cosine decay.
Tw, T = 5, 100
def lr_lambda(epoch):
    if epoch < Tw:
        return (epoch + 1) / Tw
    progress = (epoch - Tw) / max(1, T - Tw)
    return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

best_state = None
best_val = float("inf")
best_epoch = 0
patience = 10
bad_epochs = 0

history = []

for epoch in range(T):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_logits = model(Xv)
        val_ce = criterion(val_logits, yv).item()
        train_ce = criterion(model(Xtr), ytr).item()
        val_acc = (val_logits.argmax(1) == yv).float().mean().item()

    history.append((epoch, train_ce, val_ce, optimizer.param_groups[0]["lr"], val_acc))
    scheduler.step()

    if val_ce < best_val - 1e-4:
        best_val = val_ce
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        bad_epochs = 0
    else:
        bad_epochs += 1

    if epoch in [0, 4, 5, 9]:
        print(f"epoch={epoch:02d}  train_CE={train_ce:.6f}  val_CE={val_ce:.6f}  lr={optimizer.param_groups[0]['lr']:.6f}")

    if bad_epochs >= patience:
        print("Early stopping:", epoch + 1)
        break

if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
with torch.no_grad():
    test_logits = model(Xte)
    test_pred = test_logits.argmax(1).numpy()

test_acc = accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred, pos_label=0)

print("\nBEST EPOCH:", best_epoch)
print("BEST VAL CE:", f"{best_val:.6f}")
print("TEST ACCURACY:", f"{test_acc:.4f}")
print("TEST F1 (pos_label=0):", f"{test_f1:.4f}")
print("Total epochs:", len(history))
