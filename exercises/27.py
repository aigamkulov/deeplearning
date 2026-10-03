import numpy as np
import torch
import torch.nn.functional as F

def inverted_dropout(a, p, rng):
    keep = (rng.random(a.shape) >= p).astype(np.float64)
    out = a * keep / (1 - p)
    return out, keep

def inverted_dropout_backward(grad_out, mask, p):
    return grad_out * mask / (1 - p)

rng = np.random.default_rng(42)
a = rng.normal(2.0, 1.0, size=(1000, 100))
p = 0.2

out, mask = inverted_dropout(a, p, rng)
grad_out = np.ones_like(a)
grad_in = inverted_dropout_backward(grad_out, mask, p)

print("Manual inverted dropout")
print("mean(a) =", np.mean(a))
print("mean(out) =", np.mean(out))
print("mean(out)/mean(a) =", np.mean(out)/np.mean(a))
print("mean backward gradient =", np.mean(grad_in))

torch.manual_seed(42)
x = torch.ones(1000, 100, requires_grad=True)
drop = torch.nn.Dropout(p=0.2)
drop.train()
y = drop(x)
y.sum().backward()
print("\nPyTorch train mode:")
print("zero fraction =", float((y == 0).float().mean()))
print("unique nonzero values =", torch.unique(y.detach())[:5].numpy())
print("gradient unique values =", torch.unique(x.grad)[:5].numpy())

x2 = torch.ones(1000, 100, requires_grad=True)
drop.eval()
y2 = drop(x2)
print("\nPyTorch eval mode:")
print("all ones:", bool(torch.allclose(y2, torch.ones_like(y2))))
