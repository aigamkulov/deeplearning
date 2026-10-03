import torch
import torch.nn as nn

torch.manual_seed(42)

model = nn.Linear(64, 10)
criterion = nn.CrossEntropyLoss()

# Bir xil 32 ta namunadan iborat katta batch.
X = torch.randn(32, 64)
y = torch.randint(0, 10, (32,))

# (a) full batch gradient
model.zero_grad()
loss = criterion(model(X), y)
loss.backward()
full_grads = [p.grad.detach().clone() for p in model.parameters()]

# 4 ta micro-batch: loss/4 bilan yig'ish
model.zero_grad()
for xb, yb in zip(X.chunk(4), y.chunk(4)):
    micro_loss = criterion(model(xb), yb) / 4
    micro_loss.backward()
accum_grads = [p.grad.detach().clone() for p in model.parameters()]

print("Gradient accumulation:")
for i, (g_full, g_acc) in enumerate(zip(full_grads, accum_grads), 1):
    print(f"param {i}: max_abs_diff={torch.max(torch.abs(g_full-g_acc)).item():.3e}")

print("\n(b) zero_grad() siz ikki marta backward:")
model.zero_grad()
criterion(model(X[:16]), y[:16]).backward()
g1 = [p.grad.detach().clone() for p in model.parameters()]
criterion(model(X[16:]), y[16:]).backward()
g2 = [p.grad.detach().clone() for p in model.parameters()]

for i, (a, b) in enumerate(zip(g1, g2), 1):
    ratio = torch.linalg.vector_norm(b) / torch.linalg.vector_norm(a)
    print(f"param {i}: ||grad_after_two||/||grad_first|| = {ratio.item():.4f}")
