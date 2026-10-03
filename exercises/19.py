import torch

z = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)

def softmax(x):
    return torch.softmax(x, dim=0)

J = torch.autograd.functional.jacobian(softmax, z)
p = softmax(z)
J_formula = torch.diag(p) - torch.outer(p, p)

print("z =", z.numpy())
print("p =", p.detach().numpy())
print("\nJacobian J:")
print(J.detach().numpy())
print("\nFormula J = diag(p) - p p^T:")
print(J_formula.detach().numpy())

print("\n4 ta tekshiruv:")
print("1) moslik:", torch.allclose(J, J_formula, atol=1e-12).item())
print("2) simmetriya:", torch.allclose(J, J.T, atol=1e-12).item())
print("3) J @ 1 =", (J @ torch.ones(3, dtype=torch.float64)).numpy())
print("4) qator yig'indilari =", J.sum(dim=1).numpy())
