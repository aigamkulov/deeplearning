import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

X, y = load_breast_cancer(return_X_y=True)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

Xtr, Xte, ytr, yte = train_test_split(
    X.numpy(), y.numpy(), test_size=0.2, stratify=y.numpy(), random_state=42
)
Xtr, Xte = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xte, dtype=torch.float32)
ytr, yte = torch.tensor(ytr, dtype=torch.long), torch.tensor(yte, dtype=torch.long)

dataset = TensorDataset(Xtr, ytr)

def make_model():
    return nn.Sequential(nn.Linear(30, 64), nn.ReLU(), nn.Linear(64, 10))

def run_optimizer(name):
    torch.manual_seed(42)
    model = make_model()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    if name == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
    elif name == "Momentum":
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    elif name == "Adam":
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    for _ in range(20):
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

    with torch.no_grad():
        train_ce = criterion(model(Xtr), ytr).item()
        test_acc = (model(Xte).argmax(1) == yte).float().mean().item()
    return train_ce, test_acc

for name in ["SGD", "Momentum", "Adam", "AdamW"]:
    ce, acc = run_optimizer(name)
    print(f"{name:10s}  train_CE={ce:.4f}  test_accuracy={acc:.4f}")
