import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

Xb = np.column_stack([X_train, np.ones(len(X_train))])
w = np.zeros(Xb.shape[1])

rng = np.random.default_rng(42)
B_values = [8, 32, 128]

def grad_at_batch(ids):
    xb = Xb[ids]
    yb = y_train[ids]
    z = xb @ w
    p = 1 / (1 + np.exp(-z))
    return xb.T @ (p - yb) / len(ids)

# A reference full gradient at w=0.
g_full = grad_at_batch(np.arange(len(X_train)))

for B in B_values:
    estimates = []
    for _ in range(2000):
        ids = rng.integers(0, len(X_train), size=B)
        estimates.append(grad_at_batch(ids))
    estimates = np.array(estimates)

    mean = estimates.mean(axis=0)
    std = estimates.std(axis=0)
    err = np.max(np.abs(mean - g_full))

    print(f"B={B:3d}  mean_max_abs_error={err:.6f}  mean_coordinate_std={std.mean():.6f}  std*sqrt(B)={std.mean()*np.sqrt(B):.6f}")
