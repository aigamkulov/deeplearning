import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class MLP:
    def __init__(self, sizes, rng):
        self.W1 = rng.normal(0, np.sqrt(2 / sizes[0]), (sizes[0], sizes[1]))
        self.b1 = np.zeros(sizes[1])
        self.W2 = rng.normal(0, np.sqrt(2 / sizes[1]), (sizes[1], sizes[2]))
        self.b2 = np.zeros(sizes[2])

    def logits(self, X):
        h = np.maximum(0, X @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def loss_and_grad(self, X, y):
        a1 = X @ self.W1 + self.b1
        h = np.maximum(0, a1)
        z = h @ self.W2 + self.b2
        p = 1 / (1 + np.exp(-z))
        loss = np.mean(np.logaddexp(0, z) - y * z)
        dz = (p - y) / len(y)
        gW2 = h.T @ dz[:, None]
        gb2 = np.sum(dz)
        dh = dz[:, None] @ self.W2[None, :]
        da1 = dh * (a1 > 0)
        gW1 = X.T @ da1
        gb1 = da1.sum(axis=0)
        return loss, (gW1, gb1, gW2, gb2)

def accuracy(model, X, y):
    p = 1 / (1 + np.exp(-model.logits(X)))
    return np.mean((p >= 0.5) == y)

X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
y = y.astype(float)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

rng = np.random.default_rng(42)
net = MLP([X.shape[1], 64, 1], rng)
B, eta, epochs = 32, 0.1, 20

for epoch in range(1, epochs + 1):
    idx = rng.permutation(len(Xtr))
    losses = []
    for s in range(0, len(idx), B):
        ids = idx[s:s+B]
        loss, grads = net.loss_and_grad(Xtr[ids], ytr[ids])
        gW1, gb1, gW2, gb2 = grads
        net.W1 -= eta * gW1
        net.b1 -= eta * gb1
        net.W2 -= eta * gW2
        net.b2 -= eta * gb2
        losses.append(loss)

    train_acc = accuracy(net, Xtr, ytr)
    test_acc = accuracy(net, Xte, yte)
    print(f"epoch={epoch:02d}  train_CE={np.mean(losses):.4f}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")
