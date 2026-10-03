import numpy as np

def relu(x):
    return np.maximum(0.0, x)

def relu_prime(x):
    return (x > 0).astype(float)

class MLP:
    def __init__(self, sizes, act="tanh", rng=None):
        self.rng = rng or np.random.default_rng(42)
        self.act = act
        self.W1 = self.rng.normal(0, np.sqrt(1.0 / sizes[0]), (sizes[0], sizes[1]))
        self.W2 = self.rng.normal(0, np.sqrt(2.0 / sizes[1]), (sizes[1], sizes[2]))
        self.b1 = np.zeros(sizes[1])
        self.b2 = np.zeros(sizes[2])

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = np.tanh(Z1) if self.act == "tanh" else relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        self.cache = (Z1, A1, Z2)
        return Z2

    def loss(self, X, y):
        z = self.forward(X)
        return np.mean(np.logaddexp(0, z) - y * z)

    def backward(self, X, y):
        Z1, A1, Z2 = self.cache
        p = 1 / (1 + np.exp(-Z2))
        N = len(y)
        dz2 = (p - y[:, None]) / N
        gW2 = A1.T @ dz2
        gb2 = dz2.sum(axis=0)
        dA1 = dz2 @ self.W2.T
        dZ1 = dA1 * (1 - np.tanh(Z1) ** 2) if self.act == "tanh" else dA1 * relu_prime(Z1)
        gW1 = X.T @ dZ1
        gb1 = dZ1.sum(axis=0)
        return gW1, gb1, gW2, gb2

rng = np.random.default_rng(42)
X = rng.normal(size=(5, 4))
y = rng.integers(0, 3, size=5)  # demo labels for the requested gradient-check setup
y_bin = (y == 1).astype(float)

net = MLP([4, 6, 1], act="tanh", rng=np.random.default_rng(42))
loss = net.loss(X, y_bin)
gW1, gb1, gW2, gb2 = net.backward(X, y_bin)

print("Loss:", f"{loss:.6f}")
print("Gradient shapes:", gW1.shape, gb1.shape, gW2.shape, gb2.shape)

# Central-difference check for every parameter element.
eps = 1e-6
max_rel = 0.0
for name in ["W1", "b1", "W2", "b2"]:
    arr = getattr(net, name)
    analytic = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}[name]
    for idx in np.ndindex(arr.shape):
        old = arr[idx]
        arr[idx] = old + eps
        lp = net.loss(X, y_bin)
        arr[idx] = old - eps
        lm = net.loss(X, y_bin)
        arr[idx] = old
        num = (lp - lm) / (2 * eps)
        rel = abs(num - analytic[idx]) / max(abs(num), abs(analytic[idx]), 1e-12)
        max_rel = max(max_rel, rel)

print("Max relative gradient error:", f"{max_rel:.3e}")
print("OK:", max_rel < 1e-7)
