import numpy as np
import torch
import torch.nn.functional as F

Z = np.array([[1000, 1001, 1002], [-5, 0, 5]], dtype=np.float64)
y = np.array([2, 0])

def stable_log_softmax(z):
    m = np.max(z, axis=1, keepdims=True)
    return z - m - np.log(np.sum(np.exp(z - m), axis=1, keepdims=True))

def cross_entropy_mean(z, y):
    logp = stable_log_softmax(z)
    return -np.mean(logp[np.arange(len(y)), y])

logp = stable_log_softmax(Z)
loss = cross_entropy_mean(Z, y)

print("Z =")
print(Z)
print("\nlog_softmax(Z) =")
print(np.round(logp, 4))
print("\nFormula CE =", loss)

Zt = torch.tensor(Z, dtype=torch.float64)
yt = torch.tensor(y, dtype=torch.long)
torch_logp = F.log_softmax(Zt, dim=1)
torch_loss = F.cross_entropy(Zt, yt)

print("\nPyTorch log_softmax:")
print(torch.round(torch_logp, decimals=4))
print("PyTorch cross_entropy =", float(torch_loss))

print("\nBirinchi qator softmax:")
p = np.exp(logp[0])
print(np.round(p, 4))
